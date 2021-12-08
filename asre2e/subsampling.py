from typing import Optional
from typing import Tuple

import torch
from torch import Tensor
from torch import nn


class Conv2dSubsampling4(nn.Module):
    """1/4 length 下采样。

    原数据：(batch, i_num_frames, idim)
    采样后的数据：(batch, o_num_frames, odim)
    o_num_frames = ((i_num_frames - 1) // 2 - 1) // 2
    o_num_frames 大约为 i_num_frames 的 1/4

    原数据经过 conv 后：
        (
            batch,
            odim,
            ((i_num_frames - 1) // 2 - 1) // 2,
            ((idim - 1) // 2 - 1) // 2
        )
    将 1, 3 维度数据合并：
        (
            batch,
            ((i_num_frames - 1) // 2 - 1) // 2,
            odim * ((idim - 1) // 2 - 1) // 2
        )
    经过线性层后：
        (
            batch,
            ((i_num_frames - 1) // 2 - 1) // 2,
            odim
        )
    """

    def __init__(
        self,
        idim: int,
        odim: int,
        dropout_rate: float,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, odim, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(odim, odim, 3, stride=2),
            nn.ReLU(),
        )
        self.linear = nn.Sequential(
            nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim),
            nn.Dropout(p=dropout_rate),
        )

    def forward(
        self,
        xs: Tensor,
        xs_lengths: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            xs (Tensor): (batch, num_frames, idim)
            xs_lengths (Tensor): (batch,)
        Returns:
            Tensor: (batch, o_num_frames, odim)
                o_num_frames = ((num_frames - 1) // 2 - 1) // 2
        """
        # (batch, 1, num_frames, idim)
        xs = xs.unsqueeze(1)
        # (batch, odim, o_num_frames, idim_subsampled)
        # idim_subsampled = ((idim - 1) // 2 - 1) // 2
        xs = self.conv(xs)
        # (batch, o_num_frames, odim * idim_subsampled)
        xs = xs.transpose(1, 2).contiguous().view(
            xs.size(0), xs.size(2), xs.size(1) * xs.size(3))
        # (batch, o_num_frames, odim)
        xs = self.linear(xs)
        if xs_lengths is not None:
            xs_lengths = torch.div(xs_lengths - 1, 2, rounding_mode="trunc")
            xs_lengths = torch.div(xs_lengths - 1, 2, rounding_mode="trunc")
        return (xs, xs_lengths)

    def streaming(self) -> "Conv2dSubsampling4Streaming":
        return Conv2dSubsampling4Streaming(self)


class Conv2dSubsampling4Streaming:

    def __init__(self, subsampling_module: Conv2dSubsampling4):
        self.subsampling_module = subsampling_module

        self._cache = None

    def forward_chunk(self, chunk: Tensor) -> Tensor:
        """
        Args:
            chunk (Tensor): (1, num_frames, idim)
        Returns:
            Tensor: (1, o_num_frames, odim)
        """
        assert chunk.size(0) == 1
        if self._cache is not None:
            chunk = torch.cat([self._cache, chunk], dim=1)
            if chunk.size(1) < 11:  # 保证采样后至少有两帧数据。
                self._cache = chunk[:]
                return None
        next_cache_size = (chunk.size(1) - 7) % 4 + 3
        self._cache = chunk[:, -next_cache_size:, ]
        return self.subsampling_module.forward(chunk)

    def clear_cache(self) -> None:
        self._cache = None
