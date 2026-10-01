import pytest
import numpy as np

from src.neural import (
    convert_grid_to_string,
    parse_string_to_grid,
    format_qwen_chatml,
    TaskAugmenter,
    apply_d4,
    invert_d4,
    apply_color_perm,
    invert_color_perm,
)


def test_grid_string_serialization_roundtrip():
    grid = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
    ]
    serialized = convert_grid_to_string(grid)
    assert serialized == "012\n345\n678"

    recovered = parse_string_to_grid(serialized)
    assert recovered == grid


def test_parse_string_to_grid_edge_cases():
    # Ragged rows should fail
    ragged = "012\n34"
    assert parse_string_to_grid(ragged) is None

    # Empty string should fail
    assert parse_string_to_grid("") is None

    # Text with extra spaces/formatting
    messy = " 1 2 3 \n 4 5 6 "
    assert parse_string_to_grid(messy) == [[1, 2, 3], [4, 5, 6]]


def test_format_qwen_chatml():
    train_pairs = [
        {"input": [[1, 2]], "output": [[3, 4]]},
    ]
    test_in = [[5, 6]]
    test_out = [[7, 8]]

    prompt, target = format_qwen_chatml(train_pairs, test_in, test_out)
    assert "<|im_start|>user\n12<|im_end|>" in prompt
    assert "<|im_start|>assistant\n34<|im_end|>" in prompt
    assert "<|im_start|>user\n56<|im_end|>\n<|im_start|>assistant\n" in prompt
    assert target == "78<|im_end|>"


def test_d4_all_inversions():
    # Asymmetric test array
    arr = np.array([
        [1, 2, 3],
        [4, 5, 6],
    ], dtype=int)

    for i in range(8):
        transformed = apply_d4(arr, i)
        inverted = invert_d4(transformed, i)
        assert np.array_equal(inverted, arr), f"D4 inversion failed for transform_idx={i}"


def test_color_perm_inversion():
    arr = np.array([[0, 1, 9], [3, 4, 5]], dtype=int)
    perm = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

    permuted = apply_color_perm(arr, perm)
    assert permuted[0, 0] == 9
    assert permuted[0, 2] == 0

    inverted = invert_color_perm(permuted, perm)
    assert np.array_equal(inverted, arr)


def test_task_augmenter_end_to_end():
    task = {
        "train": [
            {"input": [[1, 2]], "output": [[3, 4]]},
        ],
        "test": [
            {"input": [[5, 6]], "output": [[7, 8]]},
        ],
    }

    augmenter = TaskAugmenter(rng_seed=123)
    augs = augmenter.generate_balanced_augmentations(task, num_augs=8)
    assert len(augs) == 8

    # Verify that inverting the augmented test output recovers original test output
    for aug_task, inverter in augs:
        aug_out = aug_task["test"][0]["output"]
        recovered = inverter(aug_out)
        assert recovered == [[7, 8]]
