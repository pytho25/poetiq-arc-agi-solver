# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is Poetiq's ARC-AGI solver - a system that solves Abstract Reasoning Corpus tasks by having LLMs generate Python `transform(grid)` functions. The solver uses an iterative approach where the LLM receives feedback on failed attempts and refines its solutions.

## Running the Solver

```bash
# Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure API keys in .env
GEMINI_API_KEY=...
OPENAI_API_KEY=...

# Run (modify constants in main.py first)
python main.py
```

Key constants in `main.py`:
- `DATA_CHALLENGES`: Path to challenge JSON file
- `DATA_SOLUTIONS`: Optional path to solutions for scoring
- `NUM_PROBLEMS`: Limit number of problems (None = all)
- `SELECTED_PROBLEMS`: List of specific problem IDs to run

## Architecture

### Core Solving Flow

1. **main.py** - Entry point. Loads challenges, runs `solve()` for each task concurrently, scores results, writes Kaggle-format output to `output/`

2. **solve.py** - Thin wrapper that calls `solve_parallel_coding()` with configs from `CONFIG_LIST`

3. **solve_parallel_coding.py** - Runs multiple "expert" solvers in parallel, then aggregates results via voting:
   - Groups solutions by identical test outputs (buckets)
   - Ranks by vote count (how many experts produced same output)
   - Passers (all training examples correct) ranked above failures
   - Returns ordered list of candidate solutions

4. **solve_coding.py** - Core iterative solver loop:
   - Builds prompt with problem description and previous attempts' feedback
   - Calls LLM to generate Python code
   - Parses `transform` function from markdown code block
   - Executes code in sandbox against train/test inputs
   - If all training examples pass, returns immediately
   - Otherwise, builds feedback showing diffs and continues iterations

5. **sandbox.py** - Executes generated code in isolated subprocess with stdin/stdout JSON interface. Generated code must define `transform(grid: np.ndarray) -> np.ndarray`

6. **llm.py** - LiteLLM wrapper with rate limiting, retries, and timeout handling

### Configuration

`config.py` defines `CONFIG_LIST` - list of `ExpertConfig` dicts controlling:
- Model selection (`llm_id`): Supports Gemini, OpenAI, Anthropic, xAI via litellm
- Prompt templates (`solver_prompt`, `feedback_prompt`)
- Iteration limits (`max_iterations`, `max_solutions`)
- Voting behavior (`use_new_voting`, `count_failed_matches`, `iters_tiebreak`)

Change `NUM_EXPERTS` in config.py to run multiple parallel experts (e.g., 1, 2, or 8 for different Poetiq configurations).

### Types

`types.py` defines key data structures:
- `ExpertConfig`: Solver configuration parameters
- `ARCAGIResult`: Contains train_results, test results, iteration count, token usage
- `RunResult`: Single execution result with success flag, output, soft_score, error

### Scoring

- Two attempts allowed per test input (Kaggle format)
- `scoring.py:score_task()` returns fraction of test inputs correct
- `io.py:build_kaggle_two_attempts()` extracts top 2 distinct outputs from ranked results

## Data Format

ARC tasks have train/test examples. Each example has input/output grids (2D int arrays, values 0-9). Challenges file structure:
```json
{
  "task_id": {
    "train": [{"input": [[...]], "output": [[...]]}],
    "test": [{"input": [[...]]}]
  }
}
```

## Output

Results written to `output/` directory:
- `submission_<timestamp>.json`: Kaggle format predictions
- `config_<timestamp>.json`: Configuration used
- `tokens_<timestamp>.json`: Token usage per task
