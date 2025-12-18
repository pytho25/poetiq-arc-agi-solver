#!/usr/bin/env python3
"""Interactive script to navigate ARC-AGI challenges."""

import argparse
import json
import os
import sys
from pathlib import Path

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


def print_grid(grid, label="", indent="  "):
    """Print a grid with colors."""
    if label:
        h, w = len(grid), len(grid[0]) if grid else 0
        print(f"{indent}{label} ({h}x{w}):")
    for row in grid:
        print(indent, end="")
        for cell in row:
            print(COLORS.get(cell, f"[{cell}]"), end="")
        print()
    print()


def print_grids_side_by_side(grid1, grid2, label1="Input", label2="Output"):
    """Print two grids side by side."""
    h1, w1 = len(grid1), len(grid1[0]) if grid1 else 0
    h2, w2 = len(grid2), len(grid2[0]) if grid2 else 0
    max_h = max(h1, h2)

    # Headers
    header1 = f"{label1} ({h1}x{w1})"
    header2 = f"{label2} ({h2}x{w2})"
    spacing = max(0, w1 * 2 - len(header1) + 4)
    print(f"  {header1}" + " " * spacing + f"  {header2}")

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


def clear_screen():
    """Clear the terminal screen."""
    os.system("clear" if os.name != "nt" else "cls")


def show_challenge(challenge_id, challenge, solutions=None):
    """Display a challenge with all its details."""
    clear_screen()
    print("=" * 80)
    print(f"  Challenge ID: {challenge_id}")
    print("=" * 80)
    print()

    # Training examples
    train_examples = challenge.get("train", [])
    print(f"TRAINING EXAMPLES ({len(train_examples)} pairs):")
    print("-" * 80)
    for i, example in enumerate(train_examples):
        print(f"\nExample {i + 1}:")
        print_grids_side_by_side(example["input"], example["output"])

    # Test cases
    test_cases = challenge.get("test", [])
    print(f"\nTEST INPUT(S) ({len(test_cases)} cases):")
    print("-" * 80)
    for i, test in enumerate(test_cases):
        print(f"\nTest {i + 1}:")
        print_grid(test["input"], "Input")

        # Show solution if available
        if solutions and challenge_id in solutions:
            sol = solutions[challenge_id]
            if i < len(sol):
                print_grid(sol[i], "Expected Output")

    # Color legend
    print("\nColor legend:")
    print("  ", end="")
    for i in range(10):
        print(f"{COLORS[i]}{i}", end=" ")
    print("\n")


def list_challenges(challenges, page=0, per_page=20, filter_str=""):
    """List challenges with pagination."""
    challenge_ids = sorted(challenges.keys())
    
    # Filter if search string provided
    if filter_str:
        challenge_ids = [cid for cid in challenge_ids if filter_str.lower() in cid.lower()]
    
    total = len(challenge_ids)
    total_pages = (total + per_page - 1) // per_page
    page = min(page, max(0, total_pages - 1))
    
    start_idx = page * per_page
    end_idx = min(start_idx + per_page, total)
    
    print("\n" + "=" * 80)
    print(f"  Challenges (showing {start_idx + 1}-{end_idx} of {total})")
    if filter_str:
        print(f"  Filter: '{filter_str}'")
    print("=" * 80)
    print()
    
    for i in range(start_idx, end_idx):
        cid = challenge_ids[i]
        challenge = challenges[cid]
        train_count = len(challenge.get("train", []))
        test_count = len(challenge.get("test", []))
        print(f"  [{i + 1:3d}] {cid}  (train: {train_count}, test: {test_count})")
    
    print()
    if total_pages > 1:
        print(f"  Page {page + 1}/{total_pages}  (use 'n' for next, 'p' for previous)")
    print()


def show_help():
    """Display help message."""
    print("\n" + "=" * 80)
    print("  Commands:")
    print("=" * 80)
    print("  <number>        - View challenge by list number")
    print("  <id>            - View challenge by ID (e.g., 00576224)")
    print("  list [filter]   - List all challenges (optionally filtered)")
    print("  next / n        - Next page in current list")
    print("  prev / p        - Previous page in current list")
    print("  >               - Next challenge (after viewing one)")
    print("  <               - Previous challenge (after viewing one)")
    print("  search <term>   - Search challenges by ID")
    print("  help / h        - Show this help message")
    print("  quit / q        - Exit")
    print("=" * 80)
    print()


def main():
    parser = argparse.ArgumentParser(description="Interactive ARC-AGI challenge navigator")
    parser.add_argument(
        "--dataset",
        choices=["2024", "2025"],
        default="2024",
        help="Dataset to use (default: 2024)"
    )
    parser.add_argument(
        "--split",
        choices=["training", "evaluation", "test"],
        default="evaluation",
        help="Challenge split to use (default: evaluation)"
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent.absolute()
    
    # Determine file paths based on dataset and split
    if args.split == "training":
        challenges_file = f"arc-agi_training_challenges.json"
        solutions_file = f"arc-agi_training_solutions.json"
    elif args.split == "evaluation":
        challenges_file = f"arc-agi_evaluation_challenges.json"
        solutions_file = f"arc-agi_evaluation_solutions.json"
    else:  # test
        challenges_file = f"arc-agi_test_challenges.json"
        solutions_file = None  # Test set doesn't have solutions
    
    challenges_path = script_dir / "data" / f"arc-prize-{args.dataset}" / challenges_file
    solutions_path = script_dir / "data" / f"arc-prize-{args.dataset}" / solutions_file if solutions_file else None

    # Load challenges
    if not challenges_path.exists():
        print(f"Error: Challenges file not found: {challenges_path}")
        sys.exit(1)
    
    print(f"Loading challenges from {challenges_path}...")
    with open(challenges_path, "r") as f:
        challenges = json.load(f)
    
    # Load solutions if available
    solutions = None
    if solutions_path and solutions_path.exists():
        print(f"Loading solutions from {solutions_path}...")
        with open(solutions_path, "r") as f:
            solutions = json.load(f)
    
    print(f"Loaded {len(challenges)} challenges")
    if solutions:
        print(f"Loaded solutions for {len(solutions)} challenges")
    
    # Interactive loop
    current_list = sorted(challenges.keys())
    current_page = 0
    current_filter = ""
    current_index = 0
    current_challenge_id = None  # Track currently viewed challenge
    
    show_help()
    
    while True:
        try:
            cmd = input("> ").strip().lower()
            
            if not cmd:
                continue
            
            # Parse command
            parts = cmd.split()
            command = parts[0]
            
            # Quit
            if command in ["quit", "q", "exit"]:
                print("Goodbye!")
                break
            
            # Help
            elif command in ["help", "h", "?"]:
                show_help()
            
            # List challenges
            elif command == "list":
                filter_str = " ".join(parts[1:]) if len(parts) > 1 else ""
                current_filter = filter_str
                current_page = 0
                list_challenges(challenges, current_page, filter_str=current_filter)
            
            # Search
            elif command == "search":
                if len(parts) < 2:
                    print("Usage: search <term>")
                    continue
                filter_str = " ".join(parts[1:])
                current_filter = filter_str
                current_page = 0
                list_challenges(challenges, current_page, filter_str=current_filter)
            
            # Next page
            elif command in ["next", "n"]:
                filtered = [cid for cid in sorted(challenges.keys()) 
                           if not current_filter or current_filter.lower() in cid.lower()]
                total_pages = (len(filtered) + 19) // 20
                if current_page < total_pages - 1:
                    current_page += 1
                    list_challenges(challenges, current_page, filter_str=current_filter)
                else:
                    print("Already on last page")
            
            # Previous page
            elif command in ["prev", "previous", "p"]:
                if current_page > 0:
                    current_page -= 1
                    list_challenges(challenges, current_page, filter_str=current_filter)
                else:
                    print("Already on first page")
            
            # Next challenge (sequential navigation)
            elif command == ">":
                if current_challenge_id is None:
                    print("No challenge currently viewed. Use 'list' or enter a challenge ID first.")
                    continue
                filtered = [cid for cid in sorted(challenges.keys()) 
                           if not current_filter or current_filter.lower() in cid.lower()]
                try:
                    idx = filtered.index(current_challenge_id)
                    if idx < len(filtered) - 1:
                        current_challenge_id = filtered[idx + 1]
                        show_challenge(current_challenge_id, challenges[current_challenge_id], solutions)
                    else:
                        print("Already at last challenge")
                except ValueError:
                    print("Current challenge not in filtered list")
            
            # Previous challenge (sequential navigation)
            elif command == "<":
                if current_challenge_id is None:
                    print("No challenge currently viewed. Use 'list' or enter a challenge ID first.")
                    continue
                filtered = [cid for cid in sorted(challenges.keys()) 
                           if not current_filter or current_filter.lower() in cid.lower()]
                try:
                    idx = filtered.index(current_challenge_id)
                    if idx > 0:
                        current_challenge_id = filtered[idx - 1]
                        show_challenge(current_challenge_id, challenges[current_challenge_id], solutions)
                    else:
                        print("Already at first challenge")
                except ValueError:
                    print("Current challenge not in filtered list")
            
            # View by number (from list)
            elif command.isdigit():
                num = int(command)
                filtered = [cid for cid in sorted(challenges.keys()) 
                           if not current_filter or current_filter.lower() in cid.lower()]
                if 1 <= num <= len(filtered):
                    challenge_id = filtered[num - 1]
                    current_challenge_id = challenge_id
                    show_challenge(challenge_id, challenges[challenge_id], solutions)
                else:
                    print(f"Invalid number. Range: 1-{len(filtered)}")
            
            # View by challenge ID
            else:
                # Try to find challenge by ID (case-insensitive partial match)
                potential_ids = [cid for cid in challenges.keys() if cmd.lower() in cid.lower()]
                
                if len(potential_ids) == 1:
                    challenge_id = potential_ids[0]
                    current_challenge_id = challenge_id
                    show_challenge(challenge_id, challenges[challenge_id], solutions)
                elif len(potential_ids) > 1:
                    print(f"Multiple matches found:")
                    for i, cid in enumerate(potential_ids[:10], 1):
                        print(f"  [{i}] {cid}")
                    if len(potential_ids) > 10:
                        print(f"  ... and {len(potential_ids) - 10} more")
                    print("Please be more specific or use the number from the list.")
                else:
                    print(f"Challenge '{cmd}' not found. Use 'list' to see available challenges.")
        
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except EOFError:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
