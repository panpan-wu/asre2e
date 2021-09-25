from torch import Tensor
from torch import nn

from asre2e.activation import Swish
from asre2e.mask import CaucalAttentionMaskMaker
from asre2e.conformer import ConformerEncoder
from asre2e.ctc import CTCDecoder


class ASRModel(nn.Module):

    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.ctc_loss = nn.CTCLoss(reduction="sum")

    def forward(
        self,
        xs: Tensor,
        xs_lengths: Tensor,
        ys: Tensor,
        ys_lengths: Tensor,
    ) -> Tensor:
        """
        Args:
            xs (Tensor): (batch, time, idim)
            xs_lengths (Tensor): (batch,)
            ys (Tensor): (batch, l) l 为此 batch 内最长的句子的字符数。
            ys_lengths (Tensor): (batch,)
        """
        hs, hs_lengths = self.encoder(xs, xs_lengths)
        # (batch, time_subsampled, char_size)
        ys_hat = self.decoder(hs)
        ys_hat = ys_hat.transpose(0, 1)
        loss = self.ctc_loss(ys_hat, ys, hs_lengths, ys_lengths)
        return loss


def create_asr_model(conf: dict):
    idim: int = conf.get("idim", 80)
    encoder_dim: int = conf.get("encoder_dim", 80)
    num_blocks: int = conf.get("num_blocks", 8)
    subsampling_dropout_rate: float = 0.1

    attn_num_heads: int = 8
    attn_d_head: int = encoder_dim // attn_num_heads
    attn_dropout_rate: float = 0.1

    feed_forward_dropout_rate: float = 0.1
    feed_forward_hidden_units: int = encoder_dim * 4
    feed_forward_activation: nn.Module = Swish

    conv_dropout_rate: float = 0.1
    conv_kernel_size: int = 15
    conv_activation: nn.Module = Swish
    conv_causal: bool = True

    half_step_residual: bool = True

    char_size: int = conf["char_size"]

    history_num_frames = 8
    attn_mask_maker = CaucalAttentionMaskMaker(history_num_frames)

    encoder = ConformerEncoder(
        idim=idim,
        encoder_dim=encoder_dim,
        num_blocks=num_blocks,
        subsampling_dropout_rate=subsampling_dropout_rate,
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
    encoder.set_attn_mask_maker(attn_mask_maker)
    decoder = CTCDecoder(encoder_dim, char_size)
    asr_model = ASRModel(encoder, decoder)
    return asr_model
