import math

import torch
from torch import Tensor


# 参考了 espnet
class PositionalEncoding:
    """Positional encoding.

    PE(pos, 2i)    =  sin(pos / power(10000, 2i / d_model))
    PE(pos, 2i+1)  =  cos(pos / power(10000, 2i / d_model))

    Details can be found in https://github.com/espnet/espnet/pull/2816.

    See : Appendix B in https://arxiv.org/abs/1901.02860

    Args:
        d_model (int): Embedding dimension.
        max_len (int): Maximum input length.
    """
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
    ):
        self.d_model = d_model
        self.max_len = max_len
        # [-pos, ..., 0, ..., +pos]
        self.pe: Tensor = None  # (1, 2 * max_len - 1, d_model)
        # [+pos, ..., 0, ..., -pos]
        self.pe_reversed: Tensor = None  # (1, 2 * max_len - 1, d_model)
        self._init()

    def _init(self):
        pe_positive = torch.zeros(self.max_len, self.d_model)
        pe_negative = torch.zeros(self.max_len, self.d_model)
        position = torch.arange(self.max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) *
            -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        pe_negative = pe_negative[1:]
        pe_negative = torch.flip(pe_negative, [0]).unsqueeze(0)
        pe_positive = pe_positive.unsqueeze(0)
        pe = torch.cat([pe_negative, pe_positive], dim=1)
        pe_reversed = torch.flip(pe, [1])
        self.pe = pe
        self.pe_reversed = pe_reversed

    def get_encoding(
        self,
        start: int,
        end: int,
        device: str,
        dtype: torch.dtype,
        reverse: bool = False,
    ) -> Tensor:
        assert -self.pe.size(1) <= start < end <= self.pe.size(1)

        if self.pe.dtype != dtype or self.pe.device != device:
            self.pe = self.pe.to(dtype=dtype, device=device)
            self.pe_reversed = self.pe_reversed.to(dtype=dtype, device=device)
        zero_pos_index = self.pe.size(1) // 2
        if reverse:
            rel_start = zero_pos_index - end + 1
            rel_end = zero_pos_index - start + 1
            pos = self.pe_reversed[:, rel_start:rel_end]
        else:
            rel_start = zero_pos_index + start
            rel_end = zero_pos_index + end
            pos = self.pe[:, rel_start:rel_end]
        return pos
