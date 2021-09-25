import torch
import torch.nn.functional as nnfunc
from torch import Tensor
from torch import nn

from asre2e.activation import Swish


class ConvolutionModule(nn.Module):

    def __init__(
        self,
        d_model: int,
        dropout_rate: float,
        kernel_size: int = 15,
        activation: nn.Module = Swish,
        causal: bool = True,
    ):
        """
        Args:
            d_model (int): 特征维度。
            dropout_rate (float): dropout 概率。
            kernel_size (int): 卷积核大小，必须为奇数。
            activation (nn.Module): 激活函数，默认为 Swish。
            causal (bool): 是否使用因果卷积，默认为 True。
        """
        super().__init__()
        assert kernel_size % 2 == 1
        self.causal = causal

        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, 1)

        if self.causal:
            self.depthwise_conv = CausalConvolutionModule(d_model, kernel_size)
        else:
            self.depthwise_conv = nn.Conv1d(
                d_model, d_model, kernel_size, groups=d_model,
                padding=kernel_size // 2)

        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = activation()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, 1)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, xs: Tensor) -> Tensor:
        """
        Args:
            xs: (batch, time, dim)
        Returns:
            Tensor: (batch, time, dim)
        """
        xs = self.layer_norm(xs)
        xs = xs.transpose(1, 2)  # (batch, dim, time)
        xs = self.pointwise_conv1(xs)  # (batch, 2 * dim, time)
        xs = nnfunc.glu(xs, dim=1)  # (batch, dim, time)
        xs = self.depthwise_conv(xs)  # (batch, dim, time)
        xs = self.batch_norm(xs)  # (batch, dim, time)
        xs = self.activation(xs)  # (batch, dim, time)
        xs = self.pointwise_conv2(xs)  # (batch, dim, time)
        xs = self.dropout(xs)
        xs = xs.transpose(1, 2)  # (batch, time, dim)
        return xs

    def clear_cache(self) -> None:
        if self.causal:
            self.depthwise_conv.clear_cache()

    def enable_cache(self) -> None:
        if self.causal:
            self.depthwise_conv.enable_cache()

    def disable_cache(self) -> None:
        if self.causal:
            self.depthwise_conv.disable_cache()


class CausalConvolutionModule(nn.Module):

    def __init__(
        self,
        d_model: int,
        kernel_size: int = 15,
    ):
        super().__init__()

        self.conv = nn.Conv1d(
            d_model, d_model, kernel_size, groups=d_model, padding=0)

        self._left_padding = kernel_size - 1
        self._cache = None
        self._cache_flag = False

    def forward(self, xs: Tensor) -> Tensor:
        """
        Args:
            xs (Tensor): (batch, dim, time)
        Returns:
            Tensor: (batch, dim, time)
        """
        num_frames = xs.size(2)
        if self._cache_flag:
            if self._cache is not None:
                xs = torch.cat([self._cache, xs], dim=2)
            self._cache = xs[:, :, -self._left_padding:]
        left_padding = num_frames + self._left_padding - xs.size(2)
        if left_padding > 0:
            xs = nnfunc.pad(xs, (left_padding, 0), "constant", 0.0)
        return self.conv(xs)

    def clear_cache(self) -> None:
        self._cache = None

    def enable_cache(self) -> None:
        self._cache_flag = True

    def disable_cache(self) -> None:
        self.clear_cache()
        self._cache_flag = False
