from __future__ import annotations

import pytest
import torch

from inflect_finetune.monotonic_align import maximum_path


def test_maximum_path_is_monotonic_and_respects_padding() -> None:
    scores = torch.randn(2, 8, 4)
    mask = torch.zeros_like(scores)
    mask[0, :8, :4] = 1
    mask[1, :6, :3] = 1

    path = maximum_path(scores, mask)

    for batch_index, (audio_length, text_length) in enumerate(((8, 4), (6, 3))):
        active = path[batch_index, :audio_length, :text_length]
        indices = active.argmax(dim=1)
        assert torch.all(active.sum(dim=1) == 1)
        assert indices[0].item() == 0
        assert indices[-1].item() == text_length - 1
        assert torch.all(torch.isin(indices[1:] - indices[:-1], torch.tensor([0, 1])))
        assert torch.count_nonzero(path[batch_index] * (1 - mask[batch_index])) == 0


def test_maximum_path_selects_the_highest_scoring_valid_route() -> None:
    scores = torch.full((1, 5, 3), -10.0)
    scores[0, 0, 0] = 4
    scores[0, 1, 0] = 4
    scores[0, 2, 1] = 4
    scores[0, 3, 1] = 4
    scores[0, 4, 2] = 4

    path = maximum_path(scores, torch.ones_like(scores))

    assert path[0].argmax(dim=1).tolist() == [0, 0, 1, 1, 2]


def test_maximum_path_rejects_more_tokens_than_frames() -> None:
    with pytest.raises(ValueError, match="at least one audio frame"):
        maximum_path(torch.zeros(1, 2, 3), torch.ones(1, 2, 3))
