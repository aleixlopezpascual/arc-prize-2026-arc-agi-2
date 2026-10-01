from __future__ import annotations

from typing import Any, Sequence
import numpy as np

Grid = list[list[int]]


def convert_grid_to_string(grid: Grid | np.ndarray) -> str:
    """Serializes a 2D ARC grid to a compact string of bare digits with newlines.

    Example:
        [[1, 2], [3, 4]] -> "12\\n34"
    """
    arr = np.asarray(grid, dtype=int)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D grid, got ndim={arr.ndim}")
    lines = ["".join(str(int(cell)) for cell in row) for row in arr]
    return "\n".join(lines)


def parse_string_to_grid(text: str, max_dim: int = 30) -> Grid | None:
    """Parses a serialized string back into a valid 2D rectangular grid.

    Returns:
        2D list of ints if valid, otherwise None.
    """
    if not text or not isinstance(text, str):
        return None

    lines = [line.strip() for line in text.strip().split("\n")]
    rows: list[list[int]] = []

    for line in lines:
        digits = [int(ch) for ch in line if ch.isdigit()]
        if digits:
            rows.append(digits)

    if not rows:
        return None

    # Check rectangularity
    w = len(rows[0])
    if w == 0 or w > max_dim or len(rows) > max_dim:
        return None

    if any(len(r) != w for r in rows):
        return None

    # Check cell values in [0, 9]
    for r in rows:
        if any(c < 0 or c > 9 for c in r):
            return None

    return rows


def format_qwen_chatml(
    train_pairs: list[dict[str, Any]],
    test_input: Grid | np.ndarray,
    test_output: Grid | np.ndarray | None = None,
) -> tuple[str, str | None]:
    """Formats an ARC task using Qwen ChatML template.

    Returns:
        (prompt_text, target_completion_text)
    """
    prompt_parts: list[str] = []

    # Train demonstrations
    for pair in train_pairs:
        in_str = convert_grid_to_string(pair["input"])
        out_str = convert_grid_to_string(pair["output"])
        prompt_parts.append(
            f"<|im_start|>user\n{in_str}<|im_end|>\n<|im_start|>assistant\n{out_str}<|im_end|>"
        )

    # Test input query
    test_in_str = convert_grid_to_string(test_input)
    prompt_parts.append(f"<|im_start|>user\n{test_in_str}<|im_end|>\n<|im_start|>assistant\n")

    prompt = "\n".join(prompt_parts)

    target = None
    if test_output is not None:
        target = convert_grid_to_string(test_output) + "<|im_end|>"

    return prompt, target
