import json
from typing import Any, List

from arc_agi.types import ARCAGIResult


def _coerce_grid(x: Any) -> list:
    # numpy -> list
    try:
        import numpy as _np

        if isinstance(x, _np.ndarray):
            return x.tolist()
    except Exception:
        pass
    # stringified JSON -> list
    if isinstance(x, str):
        s = x.strip()
        if s and (s[0] == "[" or s[0] == "{"):
            try:
                parsed = json.loads(s)
                return parsed
            except Exception:
                # not JSON; fall through
                return []
        else:
            return []
    # already list-like?
    if isinstance(x, list):
        return x
    return []


def build_kaggle_two_attempts(results: list[ARCAGIResult], test_in: List[List[List[int]]]):
    """
    Returns: List[{"attempt_1": grid, "attempt_2": grid}] with len == len(test_in).
    """
    num_tests = len(test_in)
    out = []

    for j in range(num_tests):
        attempts: List[list] = []

        # Sweep iterations in order; collect up to 2 successful outputs for test j
        for ar in results:
            tr = ar.get("results", [])
            if j < len(tr):
                rr = tr[j]
                grid = _coerce_grid(rr.get("output", []))
                if grid != []:
                    attempts.append(grid)
                    if len(attempts) == 2:
                        break

        # Pad with empty arrays if fewer than two attempts available
        while len(attempts) < 2:
            attempts.append([])

        out.append({"attempt_1": attempts[0], "attempt_2": attempts[1]})

    return out


def extract_codes(results: list[ARCAGIResult]) -> dict:
    """
    Extract the generated code from results.
    Returns: {"attempt_1": code, "attempt_2": code, "iteration": int}
    """
    codes: List[str] = []
    iterations: List[int] = []

    for ar in results:
        # Get code from the first test result (all test results use same code)
        tr = ar.get("results", [])
        if tr:
            code = tr[0].get("code", "")
            if code and code not in codes:
                codes.append(code)
                iterations.append(ar.get("iteration", 0))
                if len(codes) == 2:
                    break

    # Pad if fewer than 2 codes
    while len(codes) < 2:
        codes.append("")
        iterations.append(0)

    return {
        "attempt_1": codes[0],
        "attempt_2": codes[1],
        "iteration_1": iterations[0],
        "iteration_2": iterations[1],
    }
