from typing import Callable
from typing import Tuple

from torch import Tensor
from torch import nn

from asre2e.activation import Swish
from asre2e.attention import MultiHeadedSelfAttention
from asre2e.convolution import ConvolutionModule
from asre2e.feed_forward import FeedForwardModule
from asre2e.mask import make_length_mask
from asre2e.subsampling import Conv2dSubsampling4


class ConformerEncoder(nn.Module):

    def __init__(
        self,
        idim: int,
        encoder_dim: int = 256,
        num_blocks: int = 8,
        subsampling_dropout_rate: float = 0.1,
        attn_num_heads: int = 8,
        attn_d_head: int = 256 // 8,
        attn_dropout_rate: float = 0.1,
        feed_forward_dropout_rate: float = 0.1,
        feed_forward_hidden_units: int = 256 * 4,
        feed_forward_activation: nn.Module = Swish,
        conv_dropout_rate: float = 0.1,
        conv_kernel_size: int = 15,
        conv_activation: nn.Module = Swish,
        conv_causal: bool = True,
        half_step_residual: bool = True,
        global_cmvn: nn.Module = None,
    ):
        super().__init__()
        self.global_cmvn = global_cmvn
        self.subsampling = Conv2dSubsampling4(
            idim=idim,
            odim=encoder_dim,
            dropout_rate=subsampling_dropout_rate,
        )
        self.blocks = nn.ModuleList([
            ConformerBlock(
                idim=encoder_dim,
                attn_num_heads=attn_num_heads,
                attn_d_head=attn_d_head,
                attn_dropout_rate=attn_dropout_rate,
                feed_forward_dropout_rate=feed_forward_dropout_rate,
                feed_forward_hidden_units=feed_forward_hidden_units,
                feed_forward_activation=feed_forward_activation,
                conv_dropout_rate=conv_dropout_rate,
                conv_kernel_size=conv_kernel_size,
                conv_activation=conv_activation,
                conv_causal=conv_causal,
                half_step_residual=half_step_residual,
            )
            for _ in range(num_blocks)
        ])

        self._attn_mask_maker = None

    def forward(
        self,
        xs: Tensor,
        xs_lengths: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            xs (Tensor): (batch, time, dim)
            xs_lengths (Tensor): (batch,)
        Returns:
            Tuple[xs, xs_lengths]
                xs (Tensor): (batch, time_subsampled, encoder_dim)
                xs_lengths (Tensor): (batch,)
        """
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)

        # xs: (batch, time_subsampled, encoder_dim)
        xs, xs_lengths = self.subsampling(xs, xs_lengths)

        # (batch, time_subsampled)
        xs_lengths_mask = make_length_mask(xs_lengths)
        attn_mask = None
        if self._attn_mask_maker is not None:
            # (batch, time_subsampled, time_subsampled)
            attn_mask = self._attn_mask_maker(xs, xs_lengths)

        xs_lengths_non_pad_mask = ~xs_lengths_mask.unsqueeze(2)
        xs = xs.masked_fill(xs_lengths_non_pad_mask, 0.0)
        for block in self.blocks:
            xs = block(xs, attn_mask, xs_lengths_non_pad_mask)
        return (xs, xs_lengths)

    def set_attn_mask_maker(
        self,
        attn_mask_maker: Callable[[Tensor, Tensor], Tensor],
    ) -> None:
        self._attn_mask_maker = attn_mask_maker

    def streaming(self, attn_cache_size: int) -> "ConformerEncoderStreaming":
        return ConformerEncoderStreaming(self, attn_cache_size)


class ConformerEncoderStreaming:

    def __init__(self, encoder: ConformerEncoder, attn_cache_size: int):
        self.encoder = encoder
        self.attn_cache_size = attn_cache_size

        self.subsampling_streaming = self.encoder.subsampling.streaming()

        self.blocks_streaming = [
            block.streaming(self.attn_cache_size)
            for block in self.encoder.blocks
        ]
        self._attn_mask_maker = None

    def forward_chunk(self, chunk: Tensor) -> Tensor:
        assert chunk.size(0) == 1

        xs = chunk
        if self.encoder.global_cmvn is not None:
            xs = self.encoder.global_cmvn(xs)

        xs = self.subsampling_streaming.forward_chunk(xs)
        if xs is None:
            return None

        attn_mask = None
        if self._attn_mask_maker is not None:
            attn_mask = self._attn_mask_maker(xs)

        for block in self.blocks_streaming:
            xs = block.forward_chunk(xs, attn_mask)

        return xs

    def set_attn_mask_maker(
        self,
        attn_mask_maker: Callable[[Tensor, Tensor], Tensor],
    ) -> None:
        self._attn_mask_maker = attn_mask_maker

    def clear_cache(self) -> None:
        self.subsampling_streaming.clear_cache()
        for block in self.blocks_streaming:
            block.clear_cache()


class ConformerBlock(nn.Module):

    def __init__(
        self,
        idim: int,
        attn_num_heads: int,
        attn_d_head: int,
        attn_dropout_rate: float,
        feed_forward_dropout_rate: float,
        feed_forward_hidden_units: int,
        feed_forward_activation: nn.Module,
        conv_dropout_rate: float,
        conv_kernel_size: int,
        conv_activation: nn.Module,
        conv_causal: bool,
        half_step_residual: bool = True,
    ):
        super().__init__()
        if half_step_residual:
            self.feed_forward_residual_factor = 0.5
        else:
            self.feed_forward_residual_factor = 1.0

        self.feed_forward1 = FeedForwardModule(
            idim=idim,
            dropout_rate=feed_forward_dropout_rate,
            hidden_units=feed_forward_hidden_units,
        )
        self.attn = MultiHeadedSelfAttention(
            d_model=idim,
            d_head=attn_d_head,
            num_heads=attn_num_heads,
            dropout_rate=attn_dropout_rate,
        )
        self.conv = ConvolutionModule(
            d_model=idim,
            dropout_rate=conv_dropout_rate,
            kernel_size=conv_kernel_size,
            activation=conv_activation,
            causal=conv_causal,
        )
        self.feed_forward2 = FeedForwardModule(
            idim=idim,
            dropout_rate=feed_forward_dropout_rate,
            hidden_units=feed_forward_hidden_units,
        )
        self.layer_norm = nn.LayerNorm(idim)

    def forward(
        self,
        xs: Tensor,
        attn_mask: Tensor = None,
        xs_lengths_non_pad_mask: Tensor = None,
    ) -> Tensor:
        """
        Args:
            xs (Tensor): (batch, time, dim)
            attn_mask (Tensor): (batch, time, time)
            xs_lengths_non_pad_mask (Tensor): (batch, time, 1)
        Returns:
            Tensor: (batch, time, dim)
        """
        xs = xs + self.feed_forward1(xs) * self.feed_forward_residual_factor
        xs = xs + self.attn(xs, attn_mask)
        if xs_lengths_non_pad_mask is not None:
            xs = xs.masked_fill(xs_lengths_non_pad_mask, 0.0)
        xs = xs + self.conv(xs)
        if xs_lengths_non_pad_mask is not None:
            xs = xs.masked_fill(xs_lengths_non_pad_mask, 0.0)
        xs = xs + self.feed_forward2(xs) * self.feed_forward_residual_factor
        xs = self.layer_norm(xs)
        if xs_lengths_non_pad_mask is not None:
            xs = xs.masked_fill(xs_lengths_non_pad_mask, 0.0)
        return xs

    def streaming(self, attn_cache_size: int) -> "ConformerBlockStreaming":
        return ConformerBlockStreaming(self, attn_cache_size)


class ConformerBlockStreaming:

    def __init__(self, block: ConformerBlock, attn_cache_size: int):
        self.block = block
        self.attn_cache_size = attn_cache_size

        self.attn_streaming = self.block.attn.streaming(self.attn_cache_size)
        self.conv_streaming = self.block.conv.streaming()

    def forward_chunk(self, chunk: Tensor, attn_mask: Tensor) -> Tensor:
        assert chunk.size(0) == 1
        xs = chunk
        xs = xs + self.block.feed_forward1(xs) * self.block.feed_forward_residual_factor
        xs = xs + self.attn_streaming.forward_chunk(xs, attn_mask)
        xs = xs + self.conv_streaming.forward_chunk(xs)
        xs = xs + self.block.feed_forward2(xs) * self.block.feed_forward_residual_factor
        xs = self.block.layer_norm(xs)
        return xs

    def clear_cache(self) -> None:
        self.attn_streaming.clear_cache()
        self.conv_streaming.clear_cache()
