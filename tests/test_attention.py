import torch

from asre2e.attention import MultiHeadedAttention
from asre2e.attention import MultiHeadedAttentionWithRelativePos
from asre2e.attention import MultiHeadedSelfAttention
from asre2e.attention import _cal_attention
from asre2e.attention import _relative_shift


def _float_equal(a: float, b: float):
    if a < b:
        a, b = b, a
    return (a - b) < 1e-10


def test_cal_attention():
    time1 = 10
    time2 = 20
    scores = torch.ones(4, 2, time1, time2, dtype=torch.float)
    mask = torch.ones(4, time1, time2, dtype=torch.bool)
    ones = torch.ones(time1, time2, dtype=torch.bool)
    mask = mask & torch.tril(ones, diagonal=ones.size(1) - ones.size(0))
    attn = _cal_attention(scores, mask)
    attn_value = 1.0 / mask[0].sum(dim=1, dtype=torch.float)
    for i in range(time1):
        for j in range(time2):
            if mask[0][i][j]:
                assert _float_equal(attn[0][0][i][j], attn_value[i])
            else:
                assert _float_equal(attn[0][0][i][j], 0.0)


def test_relative_shift():
    time1 = 10
    time2 = 20
    start = -(time2 - 1)
    end = time1
    # [-19, -18, -17, ..., -1, 0, 1, ..., 7, 8, 9]
    pos = list(range(start, end))
    pos_matrix = torch.tensor([pos for _ in range(time1)], dtype=torch.int)
    pos_matrix = pos_matrix.unsqueeze(0).unsqueeze(0)
    assert pos_matrix.size() == (1, 1, time1, time1 + time2 - 1)
    pos_matrix_shifted = _relative_shift(pos_matrix)
    pos_matrix_right = pos_matrix_shifted[:, :, :, :time2]
    assert pos_matrix_right.size() == (1, 1, time1, time2)
    for i in range(time1):
        for j in range(time2):
            pos_right = j - (i + (time2 - time1))
            assert pos_matrix_right[0][0][i][j] == pos_right


def test_multi_headed_attention():
    d_model = 80
    d_head = 10
    num_heads = 8
    dropout_rate = 0.1
    attn_obj = MultiHeadedAttention(
        d_model=d_model,
        d_head=d_head,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
    )
    query = torch.randn(8, 100, d_model)
    key = torch.randn(8, 200, d_model)
    value = torch.randn(8, 200, d_model)
    res = attn_obj(query, key, value)
    assert res.size() == (8, 100, d_model)


def test_multi_headed_attention_with_relative_pos():
    d_model = 80
    d_head = 10
    num_heads = 8
    dropout_rate = 0.1
    attn_obj = MultiHeadedAttentionWithRelativePos(
        d_model=d_model,
        d_head=d_head,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
    )
    query = torch.randn(8, 100, d_model)
    key = torch.randn(8, 200, d_model)
    value = torch.randn(8, 200, d_model)
    res = attn_obj(query, key, value)
    assert res.size() == (8, 100, d_model)


def test_multi_headed_self_attention():
    d_model = 80
    d_head = 10
    num_heads = 8
    dropout_rate = 0.1
    attn_obj = MultiHeadedSelfAttention(
        d_model=d_model,
        d_head=d_head,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
    )
    xs = torch.randn(8, 100, d_model)
    res = attn_obj(xs)
    assert res.size() == (8, 100, d_model)

    cache_size = 20
    attn_obj.set_cache_size(cache_size)
    res = attn_obj(xs)
    res = attn_obj(xs)
    assert res.size() == (8, 100, d_model)

    mask = torch.ones(8, 100, 100 + cache_size, dtype=torch.bool)
    ones = torch.ones(100, 100 + cache_size, dtype=torch.bool)
    mask = mask & torch.tril(ones, diagonal=ones.size(1) - ones.size(0))
    res = attn_obj(xs, mask)
    assert res.size() == (8, 100, d_model)
    attn_obj.clear_cache()
    res = attn_obj(xs, mask)
    assert res.size() == (8, 100, d_model)
    res = attn_obj(xs, mask)
    assert res.size() == (8, 100, d_model)
    assert attn_obj._cache.size(1) == cache_size
