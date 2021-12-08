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
    ):
        """
        Args:
            d_model (int): 特征维度。
            dropout_rate (float): dropout 概率。
            kernel_size (int): 卷积核大小，必须为奇数。
            activation (nn.Module): 激活函数，默认为 Swish。
        """
        super().__init__()
        assert kernel_size % 2 == 1

        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, 1)
        self.depthwise_conv = CausalConvolutionModule(d_model, kernel_size)
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

    def streaming(self):
        return ConvolutionModuleStreaming(self)


class ConvolutionModuleStreaming:

    def __init__(self, conv_module: ConvolutionModule):
        self.conv_module = conv_module

        self.depthwise_conv_streaming = self.conv_module.depthwise_conv.streaming()

    def forward_chunk(self, chunk: Tensor) -> Tensor:
        assert chunk.size(0) == 1
        xs = chunk
        xs = self.conv_module.layer_norm(xs)
        xs = xs.transpose(1, 2)  # (1, dim, time)
        xs = self.conv_module.pointwise_conv1(xs)  # (1, 2 * dim, time)
        xs = nnfunc.glu(xs, dim=1)  # (1, dim, time)
        xs = self.depthwise_conv_streaming.forward_chunk(xs)  # (1, dim, time)
        xs = self.conv_module.batch_norm(xs)  # (1, dim, time)
        xs = self.conv_module.activation(xs)  # (1, dim, time)
        xs = self.conv_module.pointwise_conv2(xs)  # (1, dim, time)
        xs = self.conv_module.dropout(xs)
        xs = xs.transpose(1, 2)  # (1, time, dim)
        return xs

    def clear_cache(self) -> None:
        self.depthwise_conv_streaming.clear_cache()


class CausalConvolutionModule(nn.Module):

    def __init__(
        self,
        d_model: int,
        kernel_size: int = 15,
    ):
        super().__init__()

        self.conv = nn.Conv1d(
            d_model, d_model, kernel_size, groups=d_model, padding=0)

        self.left_padding = kernel_size - 1

    def forward(self, xs: Tensor) -> Tensor:
        """
        Args:
            xs (Tensor): (batch, dim, time)
        Returns:
            Tensor: (batch, dim, time)
        """
        xs = nnfunc.pad(xs, (self.left_padding, 0), "constant", 0.0)
        return self.conv(xs)

    def streaming(self) -> "CausalConvolutionModuleStreaming":
        return CausalConvolutionModuleStreaming(self)


class CausalConvolutionModuleStreaming:

    def __init__(self, causal_conv_module: CausalConvolutionModule):
        self.causal_conv_module = causal_conv_module

        self._left_padding = self.causal_conv_module.left_padding
        self._cache = None

    def forward_chunk(self, chunk: Tensor) -> Tensor:
        """
        Args:
            chunk (Tensor): (1, dim, time)
        Returns:
            Tensor: (1, dim, time)
        """
        assert chunk.size(0) == 1
        num_frames = chunk.size(2)
        if self._cache is not None:
            chunk = torch.cat([self._cache, chunk], dim=2)
            self._cache = chunk[:, :, -self._left_padding:]
        left_padding = num_frames + self._left_padding - chunk.size(2)
        xs = chunk
        if left_padding > 0:
            xs = nnfunc.pad(xs, (left_padding, 0), "constant", 0.0)
        xs = self.causal_conv_module.conv(xs)
        return xs

    def clear_cache(self) -> None:
        self._cache = None
