import json
import os
import re
import string
from datetime import datetime
from typing import Any, Optional

import numpy as np

from arc_agi.llm import llm
from arc_agi.prompts import WARMUP_EXPLAIN_PROMPT, WARMUP_UPDATE_PROMPT
from arc_agi.sandbox import run
from arc_agi.types import ARCAGIResult, ARCAGISolution, ExpertConfig, RunResult


def _extract_section(response: str, section_name: str) -> str:
    """Extract a section from LLM response by header name."""
    # Look for ## Section Name followed by content until next ## or end
    pattern = rf'##\s*{re.escape(section_name)}\s*\n([\s\S]*?)(?=\n##|\Z)'
    match = re.search(pattern, response, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def _extract_mermaid_diagram(response: str) -> str:
    """Extract mermaid diagram from LLM response."""
    # Look for ```mermaid ... ``` block
    pattern = r'```mermaid\s*([\s\S]*?)```'
    match = re.search(pattern, response)
    if match:
        return match.group(1).strip()
    return ""


class WarmupContext:
    """Holds the accumulated context from warmup phase."""
    def __init__(self):
        self.description: str = ""
        self.diagram: str = ""
        self.patterns: str = "- None identified yet"
        self.inconsistencies: str = "- None identified yet"
        self.questions: str = "- None yet"
        self.previous_examples: list[tuple[str, str]] = []  # List of (input_str, output_str)

    def add_example(self, input_str: str, output_str: str):
        """Add an example to the cumulative list."""
        self.previous_examples.append((input_str, output_str))

    def get_previous_examples_str(self) -> str:
        """Format previous examples as a string for the prompt."""
        if not self.previous_examples:
            return "No previous examples yet."

        parts = []
        for idx, (inp, out) in enumerate(self.previous_examples, 1):
            parts.append(f"**Example {idx}:**\n\n**Input:**\n{inp}\n\n**Output:**\n{out}")
        return "\n\n---\n\n".join(parts)

    def update_from_response(self, response: str):
        """Update context from LLM response."""
        desc = _extract_section(response, "Transformation Description")
        if desc:
            self.description = desc

        diagram = _extract_mermaid_diagram(response)
        if diagram:
            self.diagram = diagram

        patterns = _extract_section(response, "Consistent Patterns")
        if patterns:
            self.patterns = patterns

        inconsistencies = _extract_section(response, "Inconsistencies")
        if inconsistencies:
            self.inconsistencies = inconsistencies

        questions = _extract_section(response, "Open Questions")
        if questions:
            self.questions = questions

    def __str__(self) -> str:
        return f"""## Transformation Description
{self.description}

## Mermaid Diagram
```mermaid
{self.diagram}
```

## Consistent Patterns
{self.patterns}

## Inconsistencies
{self.inconsistencies}

## Open Questions
{self.questions}"""


async def _warmup_phase(
    train_in: list[list[list[int]]],
    train_out: list[list[list[int]]],
    llm_model: str,
    temperature: float,
    request_timeout: int | None,
    problem_id: str | None = None,
    verbose: bool = False,
    logs_dir: str | None = None,
) -> tuple[WarmupContext, int, int]:
    """
    Run warmup phase to help LLM understand the puzzle through progressive examples.

    Returns:
        tuple of (WarmupContext, total_prompt_tokens, total_completion_tokens)
    """
    total_prompt_tokens = 0
    total_completion_tokens = 0
    context = WarmupContext()
    warmup_logs: list[dict] = []

    num_examples = len(train_in)

    for i in range(num_examples):
        input_str = _example_to_diagram_python_list(train_in[i])
        output_str = _example_to_diagram_python_list(train_out[i])

        if i == 0:
            # First example: explain transformations and create initial understanding
            if verbose:
                print(f"    [WARMUP 1/{num_examples}] Analyzing first example")

            prompt = WARMUP_EXPLAIN_PROMPT.replace("$$input$$", input_str).replace("$$output$$", output_str)

            if verbose:
                print(f"\n{'='*40} WARMUP PROMPT {'='*40}")
                print(prompt)
                print(f"{'='*40} END WARMUP PROMPT {'='*40}\n")

            try:
                response, _, _, _, prompt_tokens, completion_tokens = await llm(
                    llm_model,
                    message=prompt,
                    temperature=temperature,
                    request_timeout=request_timeout,
                    max_remaining_time=None,
                    max_remaining_timeouts=None,
                    problem_id=problem_id,
                    retries=3,
                )
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens
                context.update_from_response(response)
                # Add this example to the cumulative list for future iterations
                context.add_example(input_str, output_str)

                # Log warmup iteration
                warmup_logs.append({
                    "step": i + 1,
                    "type": "initial_analysis",
                    "prompt": prompt,
                    "response": response,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                })

                if verbose:
                    print(f"      Tokens: {prompt_tokens} prompt + {completion_tokens} completion")
                    print(f"\n{'='*40} WARMUP RESPONSE {'='*40}")
                    print(response)
                    print(f"{'='*40} END WARMUP RESPONSE {'='*40}\n")
            except Exception as e:
                if verbose:
                    print(f"      Warmup error: {e}")
                # Still add example even if analysis failed, so we have the data
                context.add_example(input_str, output_str)

        else:
            # Subsequent examples: update understanding with new example
            if verbose:
                print(f"    [WARMUP {i+1}/{num_examples}] Updating with example {i+1}")

            # Provide defaults if extraction failed
            if not context.diagram:
                context.diagram = "flowchart TD\n    A[Input] --> B[Transform] --> C[Output]"
            if not context.description:
                context.description = "Unknown transformation"

            prompt = (WARMUP_UPDATE_PROMPT
                .replace("$$description$$", context.description)
                .replace("$$diagram$$", context.diagram)
                .replace("$$patterns$$", context.patterns)
                .replace("$$inconsistencies$$", context.inconsistencies)
                .replace("$$questions$$", context.questions)
                .replace("$$previous_examples$$", context.get_previous_examples_str())
                .replace("$$example_num$$", str(i + 1))
                .replace("$$input$$", input_str)
                .replace("$$output$$", output_str))

            if verbose:
                print(f"\n{'='*40} WARMUP PROMPT {'='*40}")
                print(prompt)
                print(f"{'='*40} END WARMUP PROMPT {'='*40}\n")

            try:
                response, _, _, _, prompt_tokens, completion_tokens = await llm(
                    llm_model,
                    message=prompt,
                    temperature=temperature,
                    request_timeout=request_timeout,
                    max_remaining_time=None,
                    max_remaining_timeouts=None,
                    problem_id=problem_id,
                    retries=3,
                )
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens
                context.update_from_response(response)
                # Add this example to the cumulative list for future iterations
                context.add_example(input_str, output_str)

                # Log warmup iteration
                warmup_logs.append({
                    "step": i + 1,
                    "type": "update_analysis",
                    "prompt": prompt,
                    "response": response,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                })

                if verbose:
                    print(f"      Tokens: {prompt_tokens} prompt + {completion_tokens} completion")
                    print(f"\n{'='*40} WARMUP RESPONSE {'='*40}")
                    print(response)
                    print(f"{'='*40} END WARMUP RESPONSE {'='*40}\n")
            except Exception as e:
                if verbose:
                    print(f"      Warmup update error: {e}")
                # Still add example even if analysis failed, so we have the data
                context.add_example(input_str, output_str)

    # Write warmup log file
    if logs_dir and problem_id and warmup_logs:
        _write_warmup_log(logs_dir, problem_id, warmup_logs, context, total_prompt_tokens, total_completion_tokens)

    if verbose:
        print(f"    [WARMUP COMPLETE] Total: {total_prompt_tokens} prompt + {total_completion_tokens} completion tokens")
        print(f"\n{'='*40} FINAL WARMUP CONTEXT {'='*40}")
        print(str(context))
        print(f"{'='*40} END WARMUP CONTEXT {'='*40}\n")

    return context, total_prompt_tokens, total_completion_tokens


async def solve_coding(
    *,
    train_in: list[list[list[int]]],
    train_out: list[list[list[int]]],
    test_in: list[list[list[int]]],
    config: ExpertConfig,
    problem_id: str | None = None,
    verbose: bool = False,
    logs_dir: str | None = None,
) -> ARCAGIResult:
    solver_prompt = config["solver_prompt"]
    feedback_prompt = config["feedback_prompt"]
    llm_model = config["llm_id"]
    max_iterations = int(config["max_iterations"])
    solver_temperature = float(config["solver_temperature"])
    max_solutions = int(config.get("max_solutions"))
    selection_probability = float(config.get("selection_probability"))
    seed = int(config.get("seed"))
    timeout_sandbox = float(config.get("timeout_s", 5))
    shuffle_examples = bool(config.get("shuffle_examples"))
    improving_order = bool(config.get("improving_order"))
    return_best = bool(config.get("return_best_result"))
    request_timeout = config.get("request_timeout")
    max_total_timeouts = config.get("max_total_timeouts")
    max_total_time = config.get("max_total_time")
    per_iteration_retries = config.get("per_iteration_retries")

    total_prompt_tokens = 0
    total_completion_tokens = 0

    best_train_score = -1.0
    best_result: Optional[ARCAGIResult] = None
    last_train: list[RunResult] = [
        RunResult(
            success=False,
            output="",
            soft_score=0.0,
            error="Unexpected use of initial empty train result",
            code="",
        )
    ]
    last_test: Optional[list[RunResult]] = None

    rng = np.random.default_rng(seed)
    solutions: list[ARCAGISolution] = []

    # Iteration logging
    iteration_logs: list[dict] = []
    start_time = datetime.now()

    # Warmup phase: progressive understanding through examples
    if verbose:
        print(f"    [WARMUP START] {problem_id} - {len(train_in)} training examples")

    warmup_context, warmup_prompt_tokens, warmup_completion_tokens = await _warmup_phase(
        train_in=train_in,
        train_out=train_out,
        llm_model=llm_model,
        temperature=solver_temperature,
        request_timeout=request_timeout,
        problem_id=problem_id,
        verbose=verbose,
        logs_dir=logs_dir,
    )
    total_prompt_tokens += warmup_prompt_tokens
    total_completion_tokens += warmup_completion_tokens

    for it in range(max_iterations):
        if verbose:
            print(f"    [ITER {it+1}/{max_iterations}] {problem_id}")

        example = _make_example(train_in, train_out, test_in)
        problem_str = format_problem(example, shuffle_examples, seed + it)

        # Build base prompt
        message = _build_prompt(solver_prompt, problem=problem_str)

        # Add warmup context at the end of the prompt
        if warmup_context and warmup_context.description:
            warmup_context_str = f"""

---

## PRE-ANALYSIS: Understanding Derived from Studying the Examples

The following analysis was derived from systematically studying each training example. Use this to guide your solution.

{str(warmup_context)}

### Guidelines for Using This Analysis
- The **Transformation Description** captures the current best understanding of the rule
- The **Mermaid Diagram** shows the algorithm flow with decision points
- **Consistent Patterns** are observations that held across all examples
- **Inconsistencies** highlight conflicts that need special handling in your code
- **Open Questions** are ambiguities your solution should account for

Your code MUST handle all identified inconsistencies and edge cases.
"""
            message = message + warmup_context_str

        selected = []
        if solutions:
            mask = rng.uniform(size=len(solutions)) < selection_probability
            selected = [s for s, keep in zip(solutions, mask, strict=False) if keep]

        if selected:
            examples_block = create_examples(
                selected, max_examples=max_solutions, improving_order=improving_order
            )
            message += "\n\n" + _build_prompt(feedback_prompt, feedback=examples_block)

        if verbose:
            print(f"      Prompt length: {len(message)} chars (~{len(message)//4} tokens)")
            if selected:
                print(f"      Including {len(selected)} previous solution(s) as feedback")
            print(f"\n{'='*40} PROMPT {'='*40}")
            print(message)
            print(f"{'='*40} END PROMPT {'='*40}\n")

        try:
            response, duration, max_total_time, max_total_timeouts, prompt_tokens, completion_tokens = await llm(
                llm_model,
                message=message,
                temperature=solver_temperature,
                request_timeout=request_timeout,
                max_remaining_time=max_total_time,
                max_remaining_timeouts=max_total_timeouts,
                problem_id=problem_id,
                retries=per_iteration_retries,
            )
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens

            if verbose:
                print(f"      LLM response: {prompt_tokens} prompt + {completion_tokens} completion tokens ({duration:.1f}s)")
                print(f"\n{'='*40} RESPONSE {'='*40}")
                print(response)
                print(f"{'='*40} END RESPONSE {'='*40}\n")

        except Exception as e:
            if verbose:
                print(f"      LLM error: {e}")
            if "Exceeded timeouts allotted to the request" in str(e) or "Exceeded time allotted to the request" in str(e):
                # Exceeded max_remaining_timeouts or max_remaining_time
                print("Exiting early due to exceeding allotted time or timeouts on problem", problem_id)
                break
            # Just exceeded per_iteration_retries, so try the next iteration
            continue

        code = _parse_code_from_llm(response)
        if not code:
            if verbose:
                print(f"      No code block found in response")
            continue

        if verbose:
            code_lines = code.strip().split('\n')
            print(f"      Generated code: {len(code_lines)} lines")

        train_res, test_res = await _eval_on_train_and_test(
            code, train_in, train_out, test_in, timeout_s=timeout_sandbox
        )

        last_train, last_test = train_res, test_res

        train_passed = sum(1 for r in train_res if r["success"])
        if verbose:
            print(f"      Train results: {train_passed}/{len(train_res)} passed")

        if all(r["success"] for r in train_res):
            if verbose:
                print(f"      SUCCESS! All train examples passed at iteration {it+1}")

            # Log the successful iteration
            if logs_dir and problem_id and code:
                code_filename = f"{problem_id}_iter_{it + 1}.py"
                with open(os.path.join(logs_dir, code_filename), "w", encoding="utf-8") as f:
                    f.write(code)
                iteration_logs.append({
                    "iteration": it + 1,
                    "timestamp": datetime.now(),
                    "prompt": message,
                    "prompt_tokens": prompt_tokens,
                    "response": response,
                    "completion_tokens": completion_tokens,
                    "duration": duration,
                    "code_file": code_filename,
                    "train_results": train_res,
                    "feedback": "All examples passed correctly.",
                    "score": 1.0,
                })
                _write_challenge_log(logs_dir, problem_id, iteration_logs, start_time,
                                     success=True, final_score=1.0)

            return ARCAGIResult(
                train_results=train_res,
                results=test_res,
                iteration=it + 1,
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
            )

        feedback, score = _build_feedback(train_res, train_in, train_out)
        solutions.append(ARCAGISolution(code=code, feedback=feedback, score=score))

        # Log this iteration
        if logs_dir and problem_id and code:
            code_filename = f"{problem_id}_iter_{it + 1}.py"
            with open(os.path.join(logs_dir, code_filename), "w", encoding="utf-8") as f:
                f.write(code)
            iteration_logs.append({
                "iteration": it + 1,
                "timestamp": datetime.now(),
                "prompt": message,
                "prompt_tokens": prompt_tokens,
                "response": response,
                "completion_tokens": completion_tokens,
                "duration": duration,
                "code_file": code_filename,
                "train_results": train_res,
                "feedback": feedback,
                "score": score,
            })

        if verbose:
            print(f"      Score: {score:.2f} (best so far: {max(best_train_score, score):.2f})")

        if score >= best_train_score:
            best_train_score = score
            best_result = ARCAGIResult(
                train_results=train_res,
                results=test_res,
                iteration=it + 1,
                prompt_tokens=None,
                completion_tokens=None,
            )

    # Write final log file if we exhausted iterations without success
    if logs_dir and problem_id and iteration_logs:
        final_score = best_train_score if best_train_score >= 0 else 0.0
        _write_challenge_log(logs_dir, problem_id, iteration_logs, start_time,
                             success=False, final_score=final_score)

    if return_best and best_result is not None:
        best_result['prompt_tokens'] = total_prompt_tokens
        best_result['completion_tokens'] = total_completion_tokens
        return best_result
    if last_test is None:
        last_test = [
            RunResult(
                success=False,
                output="",
                soft_score=0.0,
                error="Failed to generate any valid solutions.",
                code="",
            )
        ]
    return ARCAGIResult(
        train_results=last_train,
        results=last_test,
        iteration=max_iterations,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
    )


def create_examples(solutions, max_examples=3, improving_order: bool = False):
    template = string.Template("""
<solution_$index>
<solution_code>
```python
$code
```
</solution_code>
<solution_evaluation>
$feedback
</solution_evaluation>
<solution_score>
$score
</solution_score>
</solution_$index>
""")
    if not solutions:
        return ""
    scores = [x["score"] for x in solutions]
    inds = np.argsort(scores)[::-1]
    inds = inds[: min(max_examples, len(inds))]
    if improving_order:
        inds = inds[::-1]

    blocks: list[str] = []
    for k, idx in enumerate(inds, start=1):
        e = solutions[idx]
        blocks.append(
            template.substitute(
                index=k,
                code=e["code"],
                feedback=e["feedback"],
                score=f"{e['score']:.2f}",
            )
        )
    return "\n".join(blocks)


def _build_prompt(base_prompt: str, **fields: str) -> str:
    s = base_prompt
    for k, v in fields.items():
        s = s.replace(f"$${k}$$", v)
    return s


def _array_diff(arr1: np.ndarray, arr2: np.ndarray) -> str:
    rows, cols = arr1.shape
    out = []
    for i in range(rows):
        row = []
        for j in range(cols):
            if arr1[i, j] == arr2[i, j]:
                row.append(str(int(arr1[i, j])))
            else:
                row.append(f"{int(arr1[i, j])}/{int(arr2[i, j])}")
        out.append(" ".join(row))
    return "\n".join(out)


def _parse_code_from_llm(response: str) -> Optional[str]:
    m = re.search(r"```python\s*(.*?)```", response, re.DOTALL | re.IGNORECASE)
    return m.group(1) if m else None


def _soft_score(pred: np.ndarray, truth: np.ndarray) -> float:
    if pred.shape != truth.shape:
        return 0.0
    if truth.size == 0:
        return 1.0
    raw = np.mean(pred == truth)
    return float(np.nan_to_num(raw, posinf=0.0, neginf=0.0))


def _json_to_ndarray(s: str) -> Optional[np.ndarray]:
    try:
        obj = json.loads(s)
        arr = np.array(obj)
        if arr.ndim < 2:
            arr = np.expand_dims(arr, axis=list(range(2 - arr.ndim)))
        return arr.astype(int, copy=False)
    except Exception:
        return None


def _make_example(train_in, train_out, test_in) -> dict[str, Any]:
    train = [
        {"input": iin, "output": oout}
        for iin, oout in zip(train_in, train_out, strict=True)
    ]
    test = [{"input": iin} for iin in test_in]
    return {"train": train, "test": test}


def format_problem(
    problem: dict[str, Any],
    shuffle: bool = False,
    seed: Optional[int] = None,
) -> str:
    train = list(problem["train"])
    test = list(problem["test"])

    if shuffle and len(train) > 1:
        rng = np.random.default_rng(seed if seed is not None else 0)
        perm = rng.permutation(len(train))
        train = [train[i] for i in perm]

    example_str = ""
    challenge_str = ""

    for example_num, example in enumerate(train, start=1):
        example_str += f"""
Example #{example_num}
Input:
{_example_to_diagram_python_list(example["input"])}

Output:
{_example_to_diagram_python_list(example["output"])}
"""
        # Old diagram format:
        # example_str += f"""
# Example #{example_num}
# Input:
# <Diagram>
# {_example_to_diagram(example["input"])}
# </Diagram>
#
# Output:
# <Diagram>
# {_example_to_diagram(example["output"])}
# </Diagram>
# """

    for challenge_num, challenge in enumerate(test, start=1):
        challenge_str += f"""
Challenge #{challenge_num}
Input:
{_example_to_diagram_python_list(challenge["input"])}
"""
        # Old diagram format:
        # challenge_str += f"""
# Challenge #{challenge_num}
# Input:
# <Diagram>
# {_example_to_diagram(challenge["input"])}
# </Diagram>
# """

    return example_str + challenge_str


def _example_to_diagram(example: list[list[int]] | np.ndarray) -> str:
    """Converts an ARC-AGI example (list of lists) to a diagram (ascii grid)."""
    diagram = ""
    for row in example:
        row_str = " ".join([str(col) for col in row]) + "\n"
        diagram += row_str
    return diagram[:-1]  # Strip final \n


def _example_to_diagram_python_list(example: list[list[int]] | np.ndarray) -> str:
    """Converts an ARC-AGI example to a nicely formatted Python list representation."""
    lines = ["["]
    for i, row in enumerate(example):
        row_list = list(row) if hasattr(row, '__iter__') else row
        comma = "," if i < len(example) - 1 else ""
        lines.append(f"    {list(row_list)}{comma}")
    lines.append("]")
    return "\n".join(lines)


async def _eval_on_train_and_test(
    code: str,
    train_in: list[list[list[int]]],
    train_out: list[list[list[int]]],
    test_in: list[list[list[int]]],
    *,
    timeout_s: float = 1.5,
) -> tuple[list[RunResult], list[RunResult]]:
    # Train
    train_results: list[RunResult] = []
    for i, (iin, oout) in enumerate(zip(train_in, train_out, strict=True)):
        ok, out_str = await run(code, iin, timeout_s=timeout_s)
        success = False
        soft = 0.0
        err: Optional[str] = None
        if not ok:
            err = out_str or "Execution failed."
        else:
            arr = _json_to_ndarray(out_str)
            if arr is None:
                err = (
                    f"Failed to parse output as JSON 2D array.\nOutput was:\n{out_str}"
                )
            else:
                truth = np.array(oout)
                success = bool(arr.shape == truth.shape and np.array_equal(arr, truth))
                soft = _soft_score(arr, truth)
        train_results.append(
            RunResult(success=success, output=out_str, soft_score=soft, error=err, code=code)
        )

    # Test
    test_results: list[RunResult] = []
    for i, iin in enumerate(test_in):
        ok, out_str = await run(code, iin, timeout_s=timeout_s)
        err = None if ok else (out_str or "Execution failed.")
        test_results.append(
            RunResult(success=False, output=out_str, soft_score=0.0, error=err, code=code)
        )
    return train_results, test_results


def _parse_json_array_no_expand(s: str) -> Optional[np.ndarray]:
    """Parse JSON into a NumPy array without changing rank or dtype."""
    try:
        return np.array(json.loads(s))
    except Exception:
        return None


def _build_feedback(
    train_results: list[RunResult], train_in, train_out
) -> tuple[str, float]:
    feedback_parts: list[str] = []
    per_example_scores: list[float] = []

    for i, rr in enumerate(train_results):
        if rr["success"]:
            feedback_parts.append(f"Solves Example #{i + 1} correctly. ")
            per_example_scores.append(1.0)
            continue

        msg_lines: list[str] = [f"Solves Example #{i + 1} incorrectly. "]

        pred_raw = _parse_json_array_no_expand(rr["output"]) if rr["output"] else None
        truth = np.array(train_out[i])

        if pred_raw is None:
            per_example_scores.append(0.0)
            msg_lines.append("\nThe output has to be a rectangular grid of numbers.\n")
        else:
            pred_for_display = pred_raw
            if pred_for_display.ndim < 2:
                pred_for_display = np.expand_dims(
                    pred_for_display, axis=list(range(2 - pred_for_display.ndim))
                )

            if pred_raw.shape != truth.shape:
                per_example_scores.append(0.0)
                msg_lines.append(
                    f"\n\nShape mismatch: your prediction's shape was {pred_raw.shape}, "
                    f"while the correct shape was {truth.shape}."
                )
            else:
                # Same shape: show diff grid and compute soft score.
                msg_lines.append(
                    "\nYour code's output does not match the expected output."
                    "\n\nBelow is a visualization of the 2D array your code produced as well as the expected output.\n"
                    "Correctly predicted values are shown as-is while the incorrectly predicted values are shown "
                    "in the format 'prediction/correct':\n"
                )
                diff = _array_diff(pred_for_display, truth)
                msg_lines.append(f"\n```\n{diff}\n```\n")

                example_score = float(np.mean(pred_raw == truth))
                example_score = float(
                    np.nan_to_num(example_score, posinf=0.0, neginf=0.0)
                )
                per_example_scores.append(example_score)
                msg_lines.append(
                    f"Output accuracy: {example_score:.2f} (0 is worst, 1 is best).\n"
                )

        if rr["error"]:
            msg_lines.append(
                f"\n\nYour code produced the following error:\n{rr['error']}\n"
            )

        feedback_parts.append("".join(msg_lines))

    full_feedback = "\n\n".join(feedback_parts)
    mean_score = (
        float(np.mean(np.nan_to_num(per_example_scores, posinf=0.0, neginf=0.0)))
        if per_example_scores
        else 0.0
    )
    return full_feedback, mean_score


def _write_warmup_log(
    logs_dir: str,
    problem_id: str,
    warmup_logs: list[dict],
    final_context: 'WarmupContext',
    total_prompt_tokens: int,
    total_completion_tokens: int,
) -> None:
    """Write warmup phase log to TEXT file."""
    log_file = os.path.join(logs_dir, f"{problem_id}_warmup.txt")

    with open(log_file, "w", encoding="utf-8") as f:
        # Header
        f.write("=" * 80 + "\n")
        f.write(f"WARMUP PHASE: {problem_id}\n")
        f.write(f"Total Steps: {len(warmup_logs)}\n")
        f.write(f"Total Tokens: {total_prompt_tokens} prompt + {total_completion_tokens} completion\n")
        f.write("=" * 80 + "\n\n")

        # Each warmup step
        for log in warmup_logs:
            f.write("#" * 80 + "\n")
            f.write(f"WARMUP STEP {log['step']} ({log['type']})\n")
            f.write("#" * 80 + "\n\n")

            # Prompt
            f.write("-" * 80 + "\n")
            f.write(f"PROMPT ({log['prompt_tokens']} tokens)\n")
            f.write("-" * 80 + "\n")
            f.write(log['prompt'] + "\n\n")

            # Response
            f.write("-" * 80 + "\n")
            f.write(f"RESPONSE ({log['completion_tokens']} tokens)\n")
            f.write("-" * 80 + "\n")
            f.write(log['response'] + "\n\n\n")

        # Final context summary
        f.write("=" * 80 + "\n")
        f.write("FINAL WARMUP CONTEXT\n")
        f.write("=" * 80 + "\n")
        f.write(str(final_context) + "\n")
        f.write("=" * 80 + "\n")


def _write_challenge_log(
    logs_dir: str,
    problem_id: str,
    iterations: list[dict],
    start_time: datetime,
    success: bool,
    final_score: float,
) -> None:
    """Write complete challenge log to single TEXT file."""
    log_file = os.path.join(logs_dir, f"{problem_id}.txt")

    with open(log_file, "w", encoding="utf-8") as f:
        # Header
        f.write("=" * 80 + "\n")
        f.write(f"CHALLENGE: {problem_id}\n")
        f.write(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        # Each iteration
        for it in iterations:
            f.write("#" * 80 + "\n")
            f.write(f"ITERATION {it['iteration']}\n")
            f.write("#" * 80 + "\n")
            f.write(f"Timestamp: {it['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Code file: {it['code_file']}\n\n")

            # Prompt
            f.write("-" * 80 + "\n")
            f.write(f"PROMPT ({it['prompt_tokens']} tokens)\n")
            f.write("-" * 80 + "\n")
            f.write(it['prompt'] + "\n\n")

            # Response
            f.write("-" * 80 + "\n")
            f.write(f"LLM RESPONSE ({it['completion_tokens']} tokens, {it['duration']:.1f}s)\n")
            f.write("-" * 80 + "\n")
            f.write(it['response'] + "\n\n")

            # Train results
            f.write("-" * 80 + "\n")
            f.write("TRAIN RESULTS\n")
            f.write("-" * 80 + "\n")
            for i, r in enumerate(it['train_results']):
                status = "✓ PASSED" if r['success'] else "✗ FAILED"
                score_val = r.get('soft_score', 0.0)
                f.write(f"Example #{i+1}: {status} (score: {score_val:.2f})\n")
                if r.get('output'):
                    output_str = str(r['output'])
                    f.write(f"  Output: {output_str}\n")
                if r.get('error'):
                    f.write(f"  Error: {r['error']}\n")
            f.write(f"\nOverall Score: {it['score']:.2f}\n\n")

            # Feedback
            f.write("-" * 80 + "\n")
            f.write("FEEDBACK\n")
            f.write("-" * 80 + "\n")
            f.write(it['feedback'] + "\n\n\n")

        # Summary
        f.write("=" * 80 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Iterations: {len(iterations)}\n")
        f.write(f"Final Result: {'SUCCESS' if success else 'FAILED'}\n")
        f.write(f"Final Score: {final_score:.2f}\n")
        f.write("=" * 80 + "\n")
