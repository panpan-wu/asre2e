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
    ):
        super().__init__()
        self.subsampling = Conv2dSubsampling4(
            idim=idim,
            odim=encoder_dim,
            dropout_rate=subsampling_dropout_rate,
        )
        self.encoders = nn.ModuleList([
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
        for i, encoder in enumerate(self.encoders):
            xs = encoder(xs, attn_mask, xs_lengths_non_pad_mask)
        return (xs, xs_lengths)

    def set_attn_mask_maker(
        self,
        attn_mask_maker: Callable[[Tensor, Tensor], Tensor],
    ) -> None:
        self._attn_mask_maker = attn_mask_maker

    def set_cache(self, cache_size: int) -> None:
        for encoder in self.encoders:
            encoder.attn_set_cache_size(cache_size)
            encoder.conv_enable_cache()

    def clear_cache(self) -> None:
        for encoder in self.encoders:
            encoder.attn_clear_cache()
            encoder.conv_clear_cache()

    def disable_cache(self) -> None:
        for encoder in self.encoders:
            encoder.attn_set_cache_size(0)
            encoder.disable_cache()


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
        xs = xs.masked_fill(xs_lengths_non_pad_mask, 0.0)
        xs = xs + self.conv(xs)
        xs = xs.masked_fill(xs_lengths_non_pad_mask, 0.0)
        xs = xs + self.feed_forward2(xs) * self.feed_forward_residual_factor
        xs = self.layer_norm(xs)
        xs = xs.masked_fill(xs_lengths_non_pad_mask, 0.0)
        return xs

    def conv_enable_cache(self) -> None:
        self.conv.enable_cache()

    def conv_clear_cache(self) -> None:
        self.conv.clear_cache()

    def conv_disable_cache(self) -> None:
        self.conv.disable_cache()

    def attn_set_cache_size(self, cache_size: int) -> None:
        self.attn.set_cache_size(cache_size)

    def attn_clear_cache(self) -> None:
        self.attn.clear_cache()
