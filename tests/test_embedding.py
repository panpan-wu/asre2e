import torch

from asre2e.embedding import PositionalEncoding


def _float_equal(a: float, b: float):
    if a < b:
        a, b = b, a
    return (a - b) < 1e-10


def test_pos_encoding():
    d_model = 10
    max_len = 1000
    pos = PositionalEncoding(d_model, max_len)
    assert pos.pe.size(1) == 2 * max_len - 1
    assert pos.pe.size(2) == d_model

    # test reverse
    def _test_reverse(start, end):
        x = torch.tensor(0.0)
        res1 = pos.get_encoding(
            start, end, device=x.device, dtype=x.dtype, reverse=False)
        res2 = pos.get_encoding(
            start, end, device=x.device, dtype=x.dtype, reverse=True)
        num_elems = end - start
        for i in range(num_elems):
            j = num_elems - i - 1
            v1 = res1[0, i]
            v2 = res2[0, j]
            for k in range(d_model):
                assert _float_equal(v1[k], v2[k])

    _test_reverse(0, 10)
    _test_reverse(-6, 8)
