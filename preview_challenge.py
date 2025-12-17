#!/usr/bin/env python3
"""Preview an ARC-AGI challenge by task ID."""

import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile

# ANSI color codes for ARC colors (0-9)
COLORS = {
    0: "\033[40m  \033[0m",  # Black
    1: "\033[44m  \033[0m",  # Blue
    2: "\033[42m  \033[0m",  # Green
    3: "\033[43m  \033[0m",  # Yellow
    4: "\033[41m  \033[0m",  # Red
    5: "\033[45m  \033[0m",  # Magenta
    6: "\033[46m  \033[0m",  # Cyan
    7: "\033[47m  \033[0m",  # White
    8: "\033[48;5;208m  \033[0m",  # Orange
    9: "\033[48;5;52m  \033[0m",   # Brown/Maroon
}

def print_grid(grid, label=""):
    """Print a grid with colors."""
    if label:
        print(f"  {label} ({len(grid)}x{len(grid[0]) if grid else 0}):")
    for row in grid:
        print("  ", end="")
        for cell in row:
            print(COLORS.get(cell, f"[{cell}]"), end="")
        print()
    print()

def print_grid_xml(grid, tag_name):
    """Print a grid wrapped in XML tags."""
    print(f"<{tag_name}>")
    for row in grid:
        print("".join(str(cell) for cell in row))
    print(f"</{tag_name}>")
    print()


def print_grid_python(grid, var_name):
    """Print a grid as a nicely formatted Python list."""
    print(f"{var_name} = [")
    for i, row in enumerate(grid):
        comma = "," if i < len(grid) - 1 else ""
        print(f"    {list(row)}{comma}")
    print("]")
    print()

def find_latest_codes_dir(script_dir):
    """Find the most recent codes directory in output/."""
    output_dir = os.path.join(script_dir, "output")
    codes_dirs = glob.glob(os.path.join(output_dir, "codes_*"))
    codes_dirs = [d for d in codes_dirs if os.path.isdir(d)]
    if not codes_dirs:
        return None
    return max(codes_dirs, key=os.path.getmtime)


def find_latest_logs_dir(script_dir):
    """Find the most recent logs directory in output/."""
    output_dir = os.path.join(script_dir, "output")
    logs_dirs = glob.glob(os.path.join(output_dir, "logs_*"))
    logs_dirs = [d for d in logs_dirs if os.path.isdir(d)]
    if not logs_dirs:
        return None
    return max(logs_dirs, key=os.path.getmtime)


def find_latest_run_files(script_dir):
    """Find the most recent submission, tokens, and codes files."""
    output_dir = os.path.join(script_dir, "output")

    # Find the most recent submission file
    submission_files = glob.glob(os.path.join(output_dir, "submission_*.json"))
    if not submission_files:
        return None, None, None, None

    latest_submission = max(submission_files, key=os.path.getmtime)
    timestamp = os.path.basename(latest_submission).replace("submission_", "").replace(".json", "")

    tokens_file = os.path.join(output_dir, f"tokens_{timestamp}.json")
    codes_file = os.path.join(output_dir, f"codes_{timestamp}.json")
    codes_dir = os.path.join(output_dir, f"codes_{timestamp}")

    return (
        latest_submission if os.path.exists(latest_submission) else None,
        tokens_file if os.path.exists(tokens_file) else None,
        codes_file if os.path.exists(codes_file) else None,
        codes_dir if os.path.isdir(codes_dir) else None,
    )


def show_status_table(script_dir):
    """Show a table with all challenges status and run data."""
    # Load solutions
    solutions_path = os.path.join(script_dir, "data", "arc-prize-2024", "arc-agi_evaluation_solutions.json")
    solutions = {}
    if os.path.exists(solutions_path):
        with open(solutions_path, "r") as f:
            solutions = json.load(f)

    # Find latest run files
    submission_file, tokens_file, codes_file, codes_dir = find_latest_run_files(script_dir)

    if not submission_file:
        print("No submission files found in output/")
        return

    # Load data
    with open(submission_file, "r") as f:
        submissions = json.load(f)

    tokens = {}
    if tokens_file:
        with open(tokens_file, "r") as f:
            tokens = json.load(f)

    codes_info = {}
    if codes_file:
        with open(codes_file, "r") as f:
            codes_info = json.load(f)

    # Extract timestamp for display
    timestamp = os.path.basename(submission_file).replace("submission_", "").replace(".json", "")

    print(f"\n{'='*80}")
    print(f"  Run: {timestamp}")
    print(f"{'='*80}\n")

    # Table header
    print(f"{'Task ID':<12} {'Status':<10} {'Iters':<6} {'Prompt':<10} {'Completion':<12} {'Total':<10}")
    print("-" * 70)

    correct_count = 0
    total_prompt = 0
    total_completion = 0

    for task_id in sorted(submissions.keys()):
        attempts = submissions[task_id]

        # Check correctness
        status = "✗ WRONG"
        if task_id in solutions:
            gt = solutions[task_id]
            for i, pack in enumerate(attempts):
                if i < len(gt):
                    if pack.get("attempt_1") == gt[i] or pack.get("attempt_2") == gt[i]:
                        status = "✓ OK"
                        correct_count += 1
                        break
        else:
            status = "? N/A"

        # Get token info
        task_tokens = tokens.get(task_id, {})
        prompt_tokens = task_tokens.get("prompt", 0)
        completion_tokens = task_tokens.get("completion", 0)
        total_tokens = task_tokens.get("total", 0)
        total_prompt += prompt_tokens
        total_completion += completion_tokens

        # Get iteration info
        task_codes = codes_info.get(task_id, {})
        iters = task_codes.get("iteration_1", "-")

        # Format numbers with K suffix for readability
        def fmt_num(n):
            if n >= 1000000:
                return f"{n/1000000:.1f}M"
            elif n >= 1000:
                return f"{n/1000:.0f}K"
            return str(n)

        print(f"{task_id:<12} {status:<10} {str(iters):<6} {fmt_num(prompt_tokens):<10} {fmt_num(completion_tokens):<12} {fmt_num(total_tokens):<10}")

    # Summary
    print("-" * 70)
    total_tasks = len(submissions)
    print(f"\nSummary: {correct_count}/{total_tasks} correct ({100*correct_count/total_tasks:.1f}%)")
    print(f"Total tokens: {total_prompt + total_completion:,} (prompt: {total_prompt:,}, completion: {total_completion:,})")


def find_code_file(task_id, codes_dir, logs_dir=None, iteration=None):
    """Find the code file for a task.

    Args:
        task_id: The challenge ID
        codes_dir: Directory with final attempt codes (codes_*)
        logs_dir: Directory with iteration logs (logs_*)
        iteration: Specific iteration number to use (1, 2, 3, etc.)

    Returns:
        Path to the code file, or None if not found
    """
    # If specific iteration requested, look in logs directory
    if iteration is not None:
        if logs_dir and os.path.isdir(logs_dir):
            code_file = os.path.join(logs_dir, f"{task_id}_iter_{iteration}.py")
            if os.path.exists(code_file):
                return code_file
        # Also check codes_dir for iteration files (fallback)
        if codes_dir and os.path.isdir(codes_dir):
            code_file = os.path.join(codes_dir, f"{task_id}_iter_{iteration}.py")
            if os.path.exists(code_file):
                return code_file
        return None

    # Default: look for final attempt codes
    if codes_dir and os.path.isdir(codes_dir):
        for attempt in ["attempt_1", "attempt_2"]:
            code_file = os.path.join(codes_dir, f"{task_id}_{attempt}.py")
            if os.path.exists(code_file):
                return code_file
    return None


def list_available_iterations(task_id, logs_dir):
    """List available iteration files for a task."""
    if not logs_dir or not os.path.isdir(logs_dir):
        return []
    pattern = os.path.join(logs_dir, f"{task_id}_iter_*.py")
    files = glob.glob(pattern)
    iterations = []
    for f in files:
        basename = os.path.basename(f)
        # Extract iteration number from {task_id}_iter_{n}.py
        try:
            iter_num = int(basename.replace(f"{task_id}_iter_", "").replace(".py", ""))
            iterations.append(iter_num)
        except ValueError:
            pass
    return sorted(iterations)


def execute_code(code_path, input_grid, timeout=5.0):
    """Execute the transform function from a code file on an input grid.

    Uses the same execution method as arc_agi/sandbox.py
    """
    with open(code_path, "r") as f:
        code = f.read()

    # Build the execution script (same format as sandbox.py)
    script = f"""
# generated file
{code}
if __name__ == '__main__':
    import json
    import numpy as np
    import scipy
    from sys import stdin
    data = json.load(stdin)
    res = transform(np.array(data['input']))
    print(json.dumps({{"ok": True, "result": res.tolist()}}))
"""

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "u.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(script)

        try:
            proc = subprocess.run(
                [sys.executable, path],
                input=json.dumps({"input": input_grid}),
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=td,
                env={"PYTHONHASHSEED": "0"}
            )

            if proc.returncode != 0:
                return None, proc.stderr.strip() or "Execution failed"

            try:
                result = json.loads(proc.stdout)
                if result.get("ok"):
                    return result.get("result"), None
                else:
                    return None, result.get("error", "Unknown error")
            except json.JSONDecodeError:
                return None, f"Invalid output: {proc.stdout}"

        except subprocess.TimeoutExpired:
            return None, "Timeout"


def print_grids_side_by_side(grid1, grid2, label1="Input", label2="Output"):
    """Print two grids side by side."""
    h1, w1 = len(grid1), len(grid1[0]) if grid1 else 0
    h2, w2 = len(grid2), len(grid2[0]) if grid2 else 0
    max_h = max(h1, h2)

    # Headers
    print(f"  {label1} ({h1}x{w1})" + " " * (w1 * 2 - len(label1) - len(str(h1)) - len(str(w1)) + 4) + f"  {label2} ({h2}x{w2})")

    for i in range(max_h):
        # Left grid
        print("  ", end="")
        if i < h1:
            for cell in grid1[i]:
                print(COLORS.get(cell, f"[{cell}]"), end="")
        else:
            print(" " * (w1 * 2), end="")

        print("    ", end="")  # Separator

        # Right grid
        if i < h2:
            for cell in grid2[i]:
                print(COLORS.get(cell, f"[{cell}]"), end="")
        print()
    print()

def main():
    parser = argparse.ArgumentParser(description="Preview an ARC-AGI challenge")
    parser.add_argument("task_id", nargs="?", help="Task ID to preview (e.g., 00576224)")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")
    parser.add_argument("--show-solution", "-s", action="store_true", help="Show the test solution")
    parser.add_argument("--xml", "-x", action="store_true", help="Output grids in XML format (vertical, no colors)")
    parser.add_argument("--python", "-p", action="store_true", help="Output grids as formatted Python lists")
    parser.add_argument("--run-code", "-r", action="store_true", help="Run saved code and show computed output")
    parser.add_argument("--iter", "-i", type=int, help="Specific iteration number to use (e.g., -i 3 for iter_3)")
    parser.add_argument("--codes-dir", help="Path to codes directory (default: most recent in output/)")
    parser.add_argument("--logs-dir", help="Path to logs directory for iteration files (default: most recent in output/)")
    parser.add_argument("--list", "-l", action="store_true", help="Show status table of all challenges from latest run")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Show status table if --list is specified
    if args.list:
        show_status_table(script_dir)
        return

    # task_id is required if not using --list
    if not args.task_id:
        parser.error("task_id is required unless using --list")

    # Update colors to simple numbers if no-color
    if args.no_color:
        for i in range(10):
            COLORS[i] = f"{i} "

    # Load challenges
    challenges_path = os.path.join(script_dir, "data", "arc-prize-2024", "arc-agi_evaluation_challenges.json")
    solutions_path = os.path.join(script_dir, "data", "arc-prize-2024", "arc-agi_evaluation_solutions.json")

    with open(challenges_path, "r") as f:
        challenges = json.load(f)

    solutions = None
    if os.path.exists(solutions_path):
        with open(solutions_path, "r") as f:
            solutions = json.load(f)

    # Find the task
    if args.task_id not in challenges:
        print(f"Task '{args.task_id}' not found!")
        print(f"\nAvailable tasks starting with '{args.task_id[:3] if len(args.task_id) >= 3 else args.task_id}':")
        matches = [t for t in challenges.keys() if t.startswith(args.task_id[:3] if len(args.task_id) >= 3 else args.task_id)]
        for m in matches[:10]:
            print(f"  {m}")
        return

    task = challenges[args.task_id]

    # Find code file if --run-code is specified or --iter is specified
    code_file = None
    if args.run_code or args.iter is not None:
        codes_dir = args.codes_dir or find_latest_codes_dir(script_dir)
        logs_dir = args.logs_dir or find_latest_logs_dir(script_dir)

        # Show available iterations if requested iteration not found
        if args.iter is not None:
            code_file = find_code_file(args.task_id, codes_dir, logs_dir, iteration=args.iter)
            if code_file:
                print(f"Using code: {code_file}")
            else:
                available = list_available_iterations(args.task_id, logs_dir)
                if available:
                    print(f"Iteration {args.iter} not found for task {args.task_id}")
                    print(f"Available iterations: {', '.join(str(i) for i in available)}")
                else:
                    print(f"No iteration files found for task {args.task_id}")
                return
        else:
            # Default behavior: use final attempt
            if codes_dir:
                code_file = find_code_file(args.task_id, codes_dir, logs_dir)
                if code_file:
                    print(f"Using code: {code_file}")
                    # Also show available iterations for reference
                    available = list_available_iterations(args.task_id, logs_dir)
                    if available:
                        print(f"Available iterations: {', '.join(str(i) for i in available)} (use --iter N to select)")
                else:
                    print(f"No saved code found for task {args.task_id}")
            else:
                print("No codes directory found in output/")

    if args.xml:
        # XML format output
        print("<challenge>")
        print(f"<task_id>{args.task_id}</task_id>\n")

        print("<training_examples>")
        for i, example in enumerate(task["train"]):
            print(f"<example id=\"{i + 1}\">")
            print_grid_xml(example["input"], "input")
            print_grid_xml(example["output"], "output")
            print("</example>\n")
        print("</training_examples>\n")

        print("<test_cases>")
        for i, test in enumerate(task["test"]):
            print(f"<test id=\"{i + 1}\">")
            print_grid_xml(test["input"], "input")

            # Run code and show computed output
            if code_file:
                computed, error = execute_code(code_file, test["input"])
                if computed:
                    print_grid_xml(computed, "computed_output")
                else:
                    print(f"<error>{error}</error>\n")

            # Show solution if requested
            if args.show_solution and solutions and args.task_id in solutions:
                sol = solutions[args.task_id]
                if i < len(sol):
                    print_grid_xml(sol[i], "expected_output")
            print("</test>\n")
        print("</test_cases>")
        print("</challenge>")
    elif args.python:
        # Python list format output
        print(f"# Challenge: {args.task_id}")
        print(f"# Training examples: {len(task['train'])}, Test cases: {len(task['test'])}")
        print()

        print("# Training examples")
        for i, example in enumerate(task["train"]):
            print_grid_python(example["input"], f"train_{i+1}_input")
            print_grid_python(example["output"], f"train_{i+1}_output")

        print("# Test cases")
        for i, test in enumerate(task["test"]):
            print_grid_python(test["input"], f"test_{i+1}_input")

            # Run code and show computed output
            if code_file:
                computed, error = execute_code(code_file, test["input"])
                if computed:
                    print_grid_python(computed, f"test_{i+1}_computed")
                else:
                    print(f"# Error: {error}")
                    print()

            # Show solution if requested
            if args.show_solution and solutions and args.task_id in solutions:
                sol = solutions[args.task_id]
                if i < len(sol):
                    print_grid_python(sol[i], f"test_{i+1}_expected")
    else:
        # Standard colored output
        print(f"\n{'='*60}")
        print(f"  Challenge: {args.task_id}")
        print(f"{'='*60}\n")

        # Training examples
        print(f"TRAINING EXAMPLES ({len(task['train'])} pairs):")
        print("-" * 40)
        for i, example in enumerate(task["train"]):
            print(f"\nExample {i + 1}:")
            print_grids_side_by_side(example["input"], example["output"])

        # Test inputs
        print(f"\nTEST INPUT(S) ({len(task['test'])} cases):")
        print("-" * 40)
        for i, test in enumerate(task["test"]):
            print(f"\nTest {i + 1}:")
            print_grid(test["input"], "Input")

            # Run code and show computed output
            if code_file:
                computed, error = execute_code(code_file, test["input"])
                if computed:
                    print_grid(computed, "Computed Output")
                    # Check if it matches expected
                    if solutions and args.task_id in solutions:
                        sol = solutions[args.task_id]
                        if i < len(sol):
                            if computed == sol[i]:
                                print("  ✓ CORRECT\n")
                            else:
                                print("  ✗ INCORRECT\n")
                else:
                    print(f"  Error: {error}\n")

            # Show solution if requested
            if args.show_solution and solutions and args.task_id in solutions:
                sol = solutions[args.task_id]
                if i < len(sol):
                    print_grid(sol[i], "Expected Output")

        # Color legend
        print("\nColor legend:")
        print("  ", end="")
        for i in range(10):
            print(f"{COLORS[i]}{i}", end=" ")
        print("\n")

if __name__ == "__main__":
    main()
