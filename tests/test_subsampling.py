import torch

from asre2e.subsampling import Conv2dSubsampling4


def test_subsampling():
    xs = torch.randn(8, 100, 80)
    subsampling = Conv2dSubsampling4(80, 128, 0.0)
    xs_subsampled, xs_lengths = subsampling(xs, None)
    assert xs_subsampled.size() == (8, ((100 - 1) // 2 -1) // 2, 128)
    assert xs_lengths[0].item() == ((100 - 1) // 2 - 1) // 2
