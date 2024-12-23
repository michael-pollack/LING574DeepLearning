import torch
import numpy as np

import data
import run


def test_example_from_characters():
    chars = ["a", "b", "c"]
    bos = "<s>"
    eos = "</s>"
    example = data.example_from_characters(chars, bos, eos)
    assert example == {
        "text": ["<s>", "a", "b", "c"],
        "target": ["a", "b", "c", "</s>"],
        "length": 4,
    }


class TestRun:

    chars = np.array([[1, 3, 3], [1, 4, 3], [1, 4, 5]])
    padding_index = 3

    def test_get_mask(self):
        mask = run.get_mask(TestRun.chars, TestRun.padding_index)
        np.testing.assert_allclose(
            mask,
            np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]])
        )

    def test_mask_loss(self):
        loss = torch.from_numpy(np.array([
            [0.6, 0.4, 0.3],
            [0.5, 0.1, 0.4],
            [0.3, 0.2, 0.7]
        ]))
        masked_loss = run.mask_loss(loss, TestRun.chars, TestRun.padding_index)
        np.testing.assert_allclose(masked_loss.item(), 0.4)
        