import torch

from asre2e.mask import make_causal_mask
from asre2e.mask import make_length_mask


def test_make_causal_mask():
    history_num_frames = 2
    mask = make_causal_mask(10, history_num_frames, "cpu")
    for i in range(10):
        for j in range(10):
            if 0 <= i - j < history_num_frames:
                assert mask[i][j]
            else:
                assert not mask[i][j]


def test_make_length_mask():
    lengths = torch.tensor([3, 5, 8, 2], dtype=torch.int)
    mask = make_length_mask(lengths)
    assert mask.size() == (4, 8)
    for i in range(4):
        for j in range(8):
            if j < lengths[i]:
                assert mask[i][j]
            else:
                assert not mask[i][j]
