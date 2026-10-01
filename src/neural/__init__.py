"""Neural representation, tokenization, and augmentation utilities for ARC-AGI-2."""

from src.neural.augmentation import (
    TaskAugmenter,
    apply_color_perm,
    apply_d4,
    invert_color_perm,
    invert_d4,
)
from src.neural.tokenizer import (
    convert_grid_to_string,
    format_qwen_chatml,
    parse_string_to_grid,
)

__all__ = [
    "convert_grid_to_string",
    "parse_string_to_grid",
    "format_qwen_chatml",
    "TaskAugmenter",
    "apply_d4",
    "invert_d4",
    "apply_color_perm",
    "invert_color_perm",
]
