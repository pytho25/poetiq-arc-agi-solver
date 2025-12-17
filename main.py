import argparse
import asyncio
import json
import os
import resource
import time
import traceback
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv

from arc_agi.config import CONFIG_LIST
from arc_agi.io import build_kaggle_two_attempts, extract_codes
from arc_agi.scoring import score_task
from arc_agi.solve import solve

load_dotenv()

# Parse command line arguments
parser = argparse.ArgumentParser(description="ARC-AGI Solver")
parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
parser.add_argument("--log", "-L", action="store_true", help="Enable detailed iteration logging (full prompts/responses)")
parser.add_argument("--puzzle", "-p", nargs="+", help="Specific puzzle ID(s) to run (e.g., -p b7999b51 or -p id1 id2)")
args = parser.parse_args()
VERBOSE = args.verbose
LOG_ENABLED = args.log


# time the run started, so multiple runs don't collide
TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# challenge input file
DATA_CHALLENGES = os.path.join(os.path.dirname(__file__), "data", "arc-prize-2024", "arc-agi_evaluation_challenges.json")
# optional challenge solution file
DATA_SOLUTIONS = os.path.join(os.path.dirname(__file__), "data", "arc-prize-2024", "arc-agi_evaluation_solutions.json")
# where to write outputs
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
OUTPUT = os.path.join(OUTPUT_DIR, f"submission_{TIMESTAMP}.json")
LOGS_DIR = os.path.join(OUTPUT_DIR, f"logs_{TIMESTAMP}") if LOG_ENABLED else None

# number of problems (None = all)
NUM_PROBLEMS = None
# select particular problems (command line overrides this)
SELECTED_PROBLEMS = args.puzzle if args.puzzle else []  # e.g. ['b7999b51']
# max concurrent problems (for local LLM servers)
MAX_CONCURRENT = 3


async def _eval_task_data(task_id: str, task: dict, semaphore: asyncio.Semaphore, logs_dir: str | None = None) -> tuple[str, Optional[list[dict]], Optional[dict], Optional[dict], Optional[str], float]:
    """
    Returns: (task_id, kaggle_preds | None on error, tokens | None on error, codes | None on error, error, elapsed_seconds)
    """
    async with semaphore:
        start = time.time()
        if VERBOSE:
            print(f"\n{'='*60}")
            print(f"[START] Problem: {task_id}")
            train = task.get("train", [])
            test = task.get("test", [])
            print(f"  Train examples: {len(train)}")
            print(f"  Test cases: {len(test)}")
            for i, ex in enumerate(train):
                in_shape = f"{len(ex['input'])}x{len(ex['input'][0]) if ex['input'] else 0}"
                out_shape = f"{len(ex['output'])}x{len(ex['output'][0]) if ex['output'] else 0}"
                print(f"  Train #{i+1}: input {in_shape} -> output {out_shape}")
            print(f"{'='*60}")

        try:
            train = task.get("train", [])
            test = task.get("test", [])
            train_in = [ex["input"] for ex in train]
            train_out = [ex["output"] for ex in train]
            test_in = [ex["input"] for ex in test]

            results = await solve(train_in, train_out, test_in, problem_id=task_id, verbose=VERBOSE, logs_dir=logs_dir)
            kaggle_preds = build_kaggle_two_attempts(results, test_in)
            codes = extract_codes(results)

            prompt_tokens = sum(r['prompt_tokens'] or 0 for r in results if r)
            completion_tokens = sum(r['completion_tokens'] or 0 for r in results if r)
            tokens = {
                "prompt": prompt_tokens,
                "completion": completion_tokens,
                "total": prompt_tokens + completion_tokens
            }

            if VERBOSE:
                print(f"\n[DONE] Problem: {task_id}")
                print(f"  Tokens: {tokens['prompt']} prompt + {tokens['completion']} completion = {tokens['total']} total")
                print(f"  Time: {time.time() - start:.1f}s")

            return task_id, kaggle_preds, tokens, codes, None, time.time() - start
        except Exception:
            return task_id, None, None, None, traceback.format_exc(), time.time() - start


async def main():
    # Ensure we don't run out of file handles
    # Get current soft and hard limits
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    # Set a new soft limit (cannot exceed hard limit)
    new_soft = 65536
    resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

    # Create logs directory if logging is enabled
    if LOGS_DIR:
        os.makedirs(LOGS_DIR, exist_ok=True)
        print(f"Logging iterations to: {LOGS_DIR}")

    print(f"Writing config_{TIMESTAMP}.json to output directory...")
    with open(os.path.join(OUTPUT_DIR, f"config_{TIMESTAMP}.json"), "w", encoding="utf-8") as f:
        json.dump(CONFIG_LIST, f, indent=4)

    # Load challenges
    with open(DATA_CHALLENGES, "r", encoding="utf-8") as f:
        challenges_blob: dict[str, dict] = json.load(f)

    # Load solutions if present; disable scoring if missing/unreadable
    solutions_blob: Optional[dict[str, list]] = None
    if DATA_SOLUTIONS and os.path.exists(DATA_SOLUTIONS):
        try:
            with open(DATA_SOLUTIONS, "r", encoding="utf-8") as f:
                solutions_blob = json.load(f)
        except Exception as e:
            print(f"WARNING: Could not load solutions file '{DATA_SOLUTIONS}': {e}\nScoring will be disabled.")

    items = list(challenges_blob.items())
    if SELECTED_PROBLEMS:
        sel = set(SELECTED_PROBLEMS)
        items = [it for it in items if it[0] in sel]
    if NUM_PROBLEMS is not None:
        items = items[:NUM_PROBLEMS]


    print(f"Running {len(items)} problems from {DATA_CHALLENGES}...")
    print(f"Max concurrent: {MAX_CONCURRENT}")
    print(f"Verbose: {VERBOSE}")
    print(f"Iteration logging: {'enabled' if LOG_ENABLED else 'disabled'}")
    print("Scoring:", "enabled" if solutions_blob is not None else "disabled (no solutions)")
    if VERBOSE:
        print(f"\nConfig: {CONFIG_LIST[0]['llm_id']}")
        print(f"  Temperature: {CONFIG_LIST[0]['solver_temperature']}")
        print(f"  Max iterations: {CONFIG_LIST[0]['max_iterations']}")
        print(f"  Num experts: {len(CONFIG_LIST)}")

    start = time.time()

    submission: dict[str, list[dict]] = {}
    tokens_data: dict[str, dict] = {}
    codes_data: dict[str, dict] = {}

    # running scores only if solutions available
    per_task_scores: dict[str, float] = {}
    total = 0
    correct = 0.0
    incorrect = 0.0

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    tasks = [asyncio.create_task(_eval_task_data(task_id, task, semaphore, LOGS_DIR)) for task_id, task in items]

    for coro in asyncio.as_completed(tasks):
        task_id, preds, tokens, codes, err, elapsed = await coro

        if err is not None or preds is None:
            print(f"! {task_id} (error in {round(elapsed)}s)\n{err}")
            submission[task_id] = []
        else:
            submission[task_id] = preds
            if tokens:
                tokens_data[task_id] = tokens
            if codes:
                # Save code to separate files and store references
                codes_dir = os.path.join(OUTPUT_DIR, f"codes_{TIMESTAMP}")
                os.makedirs(codes_dir, exist_ok=True)

                code_refs = {}
                for attempt_key in ["attempt_1", "attempt_2"]:
                    code = codes.get(attempt_key, "")
                    if code:
                        filename = f"{task_id}_{attempt_key}.py"
                        filepath = os.path.join(codes_dir, filename)
                        with open(filepath, "w", encoding="utf-8") as cf:
                            cf.write(code)
                        code_refs[attempt_key] = filepath
                    else:
                        code_refs[attempt_key] = None

                code_refs["iteration_1"] = codes.get("iteration_1", 0)
                code_refs["iteration_2"] = codes.get("iteration_2", 0)
                codes_data[task_id] = code_refs

            # running scores if solutions available
            if solutions_blob is not None and task_id in solutions_blob:
                gt_outputs = solutions_blob[task_id]
                task_score = score_task(preds, gt_outputs)
                per_task_scores[task_id] = task_score
                total += 1
                correct += task_score
                incorrect += 1 - task_score
                mark = "✓" if task_score == 1.0 else "✗"
                print(f"{mark} {task_id} ({round(elapsed)}s) [{correct}/{total}]")
            else:
                print(f"· {task_id} ({round(elapsed)}s)")

        # write cumulative Kaggle output after each task
        try:
            with open(OUTPUT, "w", encoding="utf-8") as f:
                json.dump(submission, f)
            with open(os.path.join(OUTPUT_DIR, f"tokens_{TIMESTAMP}.json"), "w", encoding="utf-8") as f:
                json.dump(tokens_data, f)
            with open(os.path.join(OUTPUT_DIR, f"codes_{TIMESTAMP}.json"), "w", encoding="utf-8") as f:
                json.dump(codes_data, f, indent=2)
        except Exception as e:
            print(f"WARNING: Failed to write partial output to {OUTPUT}: {e}")

    total_time = time.time() - start

    print("\n=== Summary ===")
    print(f"Data file: {DATA_CHALLENGES}")
    print(f"Problems: {len(items)}")
    if solutions_blob is not None and per_task_scores:
        acc = correct / total
        print(f"Correct: {correct}")
        print(f"Incorrect: {incorrect}")
        print(f"Accuracy: {acc * 100:.3f}")
    else:
        print("Scoring: disabled or no tasks matched in solutions.")
    print(f"Total time: {round(total_time)}s")

    # final write just in case
    try:
        with open(OUTPUT, "w", encoding="utf-8") as f:
            json.dump(submission, f)
        print(f"\nWrote Kaggle submission to: {OUTPUT}")
        with open(os.path.join(OUTPUT_DIR, f"tokens_{TIMESTAMP}.json"), "w", encoding="utf-8") as f:
            json.dump(tokens_data, f)
        print(f"Wrote token usage to: {os.path.join(OUTPUT_DIR, f'tokens_{TIMESTAMP}.json')}")
        with open(os.path.join(OUTPUT_DIR, f"codes_{TIMESTAMP}.json"), "w", encoding="utf-8") as f:
            json.dump(codes_data, f, indent=2)
        print(f"Wrote generated code to: {os.path.join(OUTPUT_DIR, f'codes_{TIMESTAMP}.json')}")
    except Exception as e:
        print(f"ERROR: Final write to {OUTPUT} failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
