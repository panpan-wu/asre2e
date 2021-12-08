from typing import Tuple

import torch
from torch import Tensor
from torch import nn

from asre2e.embedding import PositionalEncoding


class MultiHeadedSelfAttention(nn.Module):
    """
    训练时设置适当的 mask 来控制能看到的过去和未来的帧数。比如一个右上三角
    全是 False，对角线和左下角全是 True 的 mask
    表示当前帧只能看到过去(包括当前)的帧，可以实现因果 attention。
    """

    def __init__(
        self,
        d_model: int,
        d_head: int,
        num_heads: int,
        dropout_rate: float = 0.1,
    ):
        """
        Args:
            d_model (int): 特征维度。
            d_head (int): 注意力头的维度。
            num_heads (int): 注意力头的数量。
            dropout_rate (float): dropout 概率。
        """
        super().__init__()
        self.attn_module = MultiHeadedAttentionWithRelativePos(
            d_model, d_head, num_heads, dropout_rate)

    def forward(self, xs: Tensor, mask: Tensor = None) -> Tensor:
        """
        Args:
            xs (Tensor): (batch, time, dim)
            mask (Tensor): (batch, time, time)
        Returns:
            Tensor: (batch, time, dim)
        """
        return self.attn_module.forward(xs, xs, xs, mask)

    def streaming(
        self,
        cache_size: int,
    ) -> "MultiHeadedSelfAttentionStreaming":
        return MultiHeadedSelfAttentionStreaming(self, cache_size)


class MultiHeadedSelfAttentionStreaming:

    def __init__(self, attn_module: MultiHeadedSelfAttention, cache_size: int):
        self.attn_module = attn_module
        self.cache_size = cache_size

        self._cache: Tensor = None

    def forward_chunk(self, chunk: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            chunk (Tensor): (1, time, dim)
            mask (Tensor): (1, time, time + cache_size)
        Returns:
            Tensor: (1, time, dim)
        """
        assert chunk.size(0) == 1
        query = chunk
        if self.cache_size > 0:
            if self._cache is None:
                key = value = query
            else:
                key = value = torch.cat([self._cache, chunk], dim=1)
            self._cache = key[:, -self.cache_size:]
            if mask is not None:
                # attention 矩阵的 shape: (batch, num_heads, time, key.size(1))
                # 当实际的缓存(self._cache)大小小于 self.cache_size 时，mask
                # 的最后一维 time + cache_size 会大于 key.size(1)，只保留 mask
                # 右边 key.size(1) 部分即可。
                mask = mask[:, :, -key.size(1):]
        else:
            key = value = query
        return self.attn_module.attn_module.forward(query, key, value, mask)

    def clear_cache(self) -> None:
        self._cache = None


class MultiHeadedAttention(nn.Module):

    def __init__(
        self,
        d_model: int,
        d_head: int,
        num_heads: int,
        dropout_rate: float,
    ):
        """
        Args:
            d_model (int): 特征维度。
            d_head (int): 注意力头的维度。
            num_heads (int): 注意力头的数量。
            dropout_rate (float): dropout 概率。
        """
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.num_heads = num_heads
        self.scale_factor = 1.0 / d_head ** 0.5

        self.linear_q = nn.Linear(d_model, num_heads * d_head, bias=False)
        self.linear_k = nn.Linear(d_model, num_heads * d_head, bias=False)
        self.linear_v = nn.Linear(d_model, num_heads * d_head, bias=False)
        self.linear_out = nn.Linear(num_heads * d_head, d_model, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor = None,
    ) -> Tensor:
        """
        Args:
            query (Tensor): (batch, time1, dim)
            key (Tensor): (batch, time2, dim)
            value (Tensor): (batch, time2, dim)
            mask (Tensor): (batch, time1, time2)
        Returns:
            Tensor: (batch, time1, dim)
        """
        q, k, v = self._forward_qkv(query, key, value)
        # (batch, num_heads, time1, time2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale_factor
        attn = _cal_attention(scores, mask)
        attn = self.dropout(attn)
        # (batch, num_heads, time1, d_head)
        z = torch.matmul(attn, v)
        # (batch, time1, num_heads * d_head)
        z = z.transpose(1, 2).contiguous().view(
            query.size(0), -1, self.num_heads * self.d_head)
        # (batch, time1, dim)
        return self.linear_out(z)

    def _forward_qkv(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            query (Tensor): (batch, time1, dim)
            key (Tensor): (batch, time2, dim)
            value (Tensor): (batch, time2, dim)
        Returns:
            Tuple[q, k, v]
                q (Tensor): (batch, num_heads, time1, d_head)
                k (Tensor): (batch, num_heads, time2, d_head)
                v (Tensor): (batch, num_heads, time2, d_head)
        """
        batch_size = query.size(0)
        q = self.linear_q(query).view(
            batch_size, -1, self.num_heads, self.d_head)
        k = self.linear_k(key).view(
            batch_size, -1, self.num_heads, self.d_head)
        v = self.linear_v(value).view(
            batch_size, -1, self.num_heads, self.d_head)
        q = q.transpose(1, 2)  # (batch, num_heads, time1, d_head)
        k = k.transpose(1, 2)  # (batch, num_heads, time2, d_head)
        v = v.transpose(1, 2)  # (batch, num_heads, time2, d_head)
        return (q, k, v)


class MultiHeadedAttentionWithRelativePos(nn.Module):

    def __init__(
        self,
        d_model: int,
        d_head: int,
        num_heads: int,
        dropout_rate: float,
    ):
        """
        Args:
            d_model (int): 特征维度。
            d_head (int): 注意力头的维度。
            num_heads (int): 注意力头的数量。
            dropout_rate (float): dropout 概率。
        """
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.num_heads = num_heads
        self.scale_factor = 1.0 / d_head ** 0.5

        self.linear_q = nn.Linear(d_model, num_heads * d_head, bias=False)
        self.linear_k = nn.Linear(d_model, num_heads * d_head, bias=False)
        self.linear_v = nn.Linear(d_model, num_heads * d_head, bias=False)
        self.linear_pos = nn.Linear(d_model, num_heads * d_head, bias=False)
        self.linear_out = nn.Linear(num_heads * d_head, d_model, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        nn.init.xavier_uniform_(self.u_bias)
        nn.init.xavier_uniform_(self.v_bias)

        self.pos_enc_obj = PositionalEncoding(d_model)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor = None,
    ) -> Tensor:
        """
        Args:
            query (Tensor): (batch, time1, dim)
            key (Tensor): (batch, time2, dim)
            value (Tensor): (batch, time2, dim)
            mask (Tensor): (batch, time1, time2)
        Returns:
            Tensor: (batch, time1, dim)
        """
        q, k, v = self._forward_qkv(query, key, value)
        pos_enc = self._get_pos(query, key)  # (1, time1 + time2 - 1, dim)
        # (batch, num_heads, time1, time2)
        scores = self._cal_scores(q, k, v, pos_enc)
        attn = _cal_attention(scores, mask)
        attn = self.dropout(attn)
        # (batch, num_heads, time1, d_head)
        z = torch.matmul(attn, v)
        # (batch, time1, num_heads * d_head)
        z = z.transpose(1, 2).contiguous().view(
            query.size(0), -1, self.num_heads * self.d_head)
        # (batch, time1, dim)
        return self.linear_out(z)

    def _cal_scores(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        pos_enc: Tensor,
    ) -> Tensor:
        """
        Args:
            q (Tensor): (batch, num_heads, time1, d_head)
            k (Tensor): (batch, num_heads, time2, d_head)
            v (Tensor): (batch, num_heads, time2, d_head)
            pos_enc (Tensor): (1, time1 + time2 - 1, dim)
        Returns:
            Tensor: (batch, num_heads, time1, time2)
        """
        time2 = k.size(2)
        # (1, time1 + time2 - 1, num_heads, d_head)
        p = self.linear_pos(pos_enc).view(1, -1, self.num_heads, self.d_head)
        # (1, num_heads, time1 + time2 - 1, d_head)
        p = p.transpose(1, 2)

        # (batch, time1, num_heads, d_head)
        q = q.transpose(1, 2)
        # (batch, num_heads, time1, d_head)
        q_with_u = (q + self.u_bias).transpose(1, 2)
        # (batch, num_heads, time1, d_head)
        q_with_v = (q + self.v_bias).transpose(1, 2)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, num_heads, time1, time2)
        matrix_ac = torch.matmul(q_with_u, k.transpose(-2, -1))

        # compute matrix b and matrix d
        # (batch, num_heads, time1, time1 + time2 - 1)
        matrix_bd = torch.matmul(q_with_v, p.transpose(-2, -1))
        # (batch, num_heads, time1, time1 + time2 - 1)
        matrix_bd = _relative_shift(matrix_bd)
        # (batch, num_heads, time1, time2)
        matrix_bd = matrix_bd[:, :, :, :time2]

        # (batch, num_heads, time1, time2)
        scores = (matrix_ac + matrix_bd) * self.scale_factor
        return scores

    def _forward_qkv(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            query (Tensor): (batch, time1, dim)
            key (Tensor): (batch, time2, dim)
            value (Tensor): (batch, time2, dim)
        Returns:
            Tuple[q, k, v]
                q (Tensor): (batch, num_heads, time1, d_head)
                k (Tensor): (batch, num_heads, time2, d_head)
                v (Tensor): (batch, num_heads, time2, d_head)
        """
        batch_size = query.size(0)
        q = self.linear_q(query).view(
            batch_size, -1, self.num_heads, self.d_head)
        k = self.linear_k(key).view(
            batch_size, -1, self.num_heads, self.d_head)
        v = self.linear_v(value).view(
            batch_size, -1, self.num_heads, self.d_head)
        q = q.transpose(1, 2)  # (batch, num_heads, time1, d_head)
        k = k.transpose(1, 2)  # (batch, num_heads, time2, d_head)
        v = v.transpose(1, 2)  # (batch, num_heads, time2, d_head)
        return (q, k, v)

    def _get_pos(self, query: Tensor, key: Tensor) -> Tensor:
        """得到位置编码。

        用 P 表示位置的话，得到的编码大概如下：
            [P-(time2-1), P-(time2-2), ..., P-1, P0, P1, ..., Ptime1-1]

        Args:
            query (Tensor): (batch, time1, dim)
            key (Tensor): (batch, time2, dim)
        Returns:
            Tensor: (1, time1 + time2 - 1, dim)
        """
        start = -(key.size(1) - 1)
        end = query.size(1)
        return self.pos_enc_obj.get_encoding(
            start=start,
            end=end,
            device=query.device,
            dtype=query.dtype,
            reverse=False,
        )


def _cal_attention(
    scores: Tensor,
    mask: Tensor = None,
) -> Tensor:
    """
    Args:
        scores (Tensor): (batch, num_heads, time1, time2)
        mask (Tensor): (batch, time1, time2)
    Returns:
        Tensor: (batch, num_heads, time1, time2)
    """
    if mask is not None:
        mask = mask.unsqueeze(1).eq(0)  # (batch, 1, time1, time2)
        # power(e, -float("inf")) 等于 0
        scores = scores.masked_fill(mask, -float("inf"))
        # 如果某一行全是 -float("inf"), 计算 softmax 后会得到 nan，
        # 故需要再次 mask fill。
        attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)
    else:
        attn = torch.softmax(scores, dim=-1)
    return attn


def _relative_shift(x: Tensor) -> Tensor:
    """将位置编码移动到正确的位置。

    Args:
        x (Tensor): (batch, num_heads, size1, size2)
    Returns:
        Tensor: (batch, num_heads, size1, size2)
    """
    zero_pad = torch.zeros(
        (x.size(0), x.size(1), x.size(2), 1),
        device=x.device,
        dtype=x.dtype,
    )
    x_padded = torch.cat([zero_pad, x], dim=-1)
    x_padded = x_padded.view(
        x.size(0), x.size(1), x.size(3) + 1, x.size(2))
    x = x_padded[:, :, 1:].view_as(x)
    return x
