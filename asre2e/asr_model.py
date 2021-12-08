from typing import List
from typing import Tuple

import torch
from torch import Tensor
from torch import nn

from asre2e.activation import Swish
from asre2e.cmvn import GlobalCMVN
from asre2e.cmvn import load_cmvn
from asre2e.conformer import ConformerEncoder
from asre2e.ctc import CTCDecoder
from asre2e.ctc import ctc_beam_search
from asre2e.ctc import ctc_greedy_search
from asre2e.ctc import ctc_merge_duplicates_and_remove_blanks
from asre2e.ctc import ctc_prefix_beam_search
from asre2e.mask import CausalAttentionMaskMaker
from asre2e.mask import StreamingAttentionMaskMaker


class SearchType:
    ctc_prefix_beam_search = "ctc_prefix_beam_search"
    ctc_beam_search = "ctc_beam_search"
    ctc_greedy_search = "ctc_greedy_search"


class BaseStreamingRecognizer:
    """流式识别。

    开始识别一段新的语音前，需要调用 clear_cache 清除掉缓存。

    示例：
        recognizer = asr_model.streaming_recognizer(
            search_type=SearchType.ctc_prefix_beam_search,
            cache_size=16,
            beam_size=2,
        )
        recognizer.clear_cache()
        chunk = torch.randn(chunk_size, dim)
        res = recognizer.forward_chunk(chunk)
    """

    def __init__(
        self,
        asr_model_streaming: nn.Module,
        beam_size: int = 1,
        blank_id: int = 0,
    ):
        self.asr_model_streaming = asr_model_streaming
        self.beam_size = beam_size
        self.blank_id = blank_id

        self._prev_beam = None

    def forward_chunk(self, chunk: Tensor) -> List[List[int]]:
        """处理部分语音，用以支持流式识别。

        子类需要覆写此方法来支持不同的识别算法。

        Args:
            chunk (Tensor): (time, dim)
        Returns:
            List[List[int]]:
                [
                    [char_id, ...],
                    ...
                ]
        """
        raise NotImplementedError()

    def forward(
        self,
        xs: Tensor,
        xs_lengths: Tensor,
        chunk_size: int,
    ) -> List[List[List[int]]]:
        """
        Args:
            xs (Tensor): (batch, time, dim)
            xs_lengths (Tensor): (batch,)
            chunk_size (int): chunk 大小。
        Returns:
            List[List[List[int]]]:
                [
                    [
                        [char_id, ...],  # beam 1
                        [char_id, ...],  # beam 2
                        ...
                    ],  # batch 1
                    [
                        [char_id, ...],  # beam 1
                        [char_id, ...],  # beam 2
                        ...
                    ], # batch 2
                    ...
                ]
        """
        batch_size = xs.size(0)
        res = []
        for batch_idx in range(batch_size):
            x_length = xs_lengths[batch_idx].item()
            x = xs[batch_idx][:x_length]
            num_frames = x.size(0)
            self.clear_cache()
            for start in range(0, num_frames, chunk_size):
                chunk = x[start:start + chunk_size]
                beam = self.forward_chunk(chunk)
            res.append(beam)
        return res

    def clear_cache(self) -> None:
        """清除缓存。

        开始识别一段新的语音前，需要调用此方法清除掉缓存。
        """
        self.asr_model_streaming.clear_cache()
        self._prev_beam = None


class CTCPrefixBeamSearchStreamingRecognizer(BaseStreamingRecognizer):

    def forward_chunk(self, chunk: Tensor) -> List[List[int]]:
        ys_hat = self.asr_model_streaming.forward_chunk(chunk)
        self._prev_beam: List[Tuple[List[int], Tuple[float, float]]] = (
            ctc_prefix_beam_search(
                ys_hat, self.beam_size, self._prev_beam, self.blank_id))
        char_ids = [e[0] for e in self._prev_beam]
        return char_ids


class CTCBeamSearchStreamingRecognizer(BaseStreamingRecognizer):

    def forward_chunk(self, chunk: Tensor) -> List[List[int]]:
        ys_hat = self.asr_model_streaming.forward_chunk(chunk)
        self._prev_beam: List[Tuple[List[int], float]] = (
            ctc_beam_search(ys_hat, self.beam_size, self._prev_beam))
        char_ids = [
            ctc_merge_duplicates_and_remove_blanks(e[0], self.blank_id)
            for e in self._prev_beam
        ]
        return char_ids


class CTCGreedySearchStreamingRecognizer(BaseStreamingRecognizer):

    def forward_chunk(self, chunk: Tensor) -> List[List[int]]:
        ys_hat = self.asr_model_streaming.forward_chunk(chunk)
        char_ids: List[int] = ctc_greedy_search(ys_hat)
        char_ids = [ctc_merge_duplicates_and_remove_blanks(char_ids)]
        return char_ids


class ASRModel(nn.Module):

    def __init__(self, encoder: nn.Module, ctc_decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.ctc_decoder = ctc_decoder
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
        Returns:
            Tensor: ctc loss.
        """
        hs, xs_lengths = self.encoder(xs, xs_lengths)
        # (batch, time_subsampled, char_size)
        ys_hat = self.ctc_decoder(hs)
        ys_hat = ys_hat.transpose(0, 1)
        loss = self.ctc_loss(ys_hat, ys, xs_lengths, ys_lengths)
        return loss

    def get_ctc_log_probs(self, xs: Tensor, xs_lengths: Tensor):
        hs, xs_lengths = self.encoder(xs, xs_lengths)
        ys_hat = self.ctc_decoder(hs)
        return (ys_hat, xs_lengths)

    def ctc_search(
        self,
        xs: Tensor,
        xs_lengths: Tensor,
        search_type: str,
        beam_size: int = 1,
        blank_id: int = 0,
    ) -> List[List[List[int]]]:
        hs, xs_lengths = self.encoder(xs, xs_lengths)
        ys_hat = self.ctc_decoder(hs)
        batch_size = ys_hat.size(0)
        res = []
        for i in range(batch_size):
            j = xs_lengths[i].item()
            char_ids: List[List[int]] = None
            inputs = ys_hat[i, :j]
            if search_type == SearchType.ctc_greedy_search:
                char_ids = [ctc_greedy_search(inputs)]
            elif search_type == SearchType.ctc_beam_search:
                char_id_probs: List[Tuple[List[int], float]] = (
                    ctc_beam_search(inputs, beam_size))
                char_ids = [e[0] for e in char_id_probs]
            elif search_type == SearchType.ctc_prefix_beam_search:
                char_id_probs: List[Tuple[List[int], Tuple[float, float]]] = (
                    ctc_prefix_beam_search(inputs, beam_size))
                char_ids = [e[0] for e in char_id_probs]
            else:
                raise ValueError("unknown search type: %s" % search_type)
            if (
                search_type == SearchType.ctc_greedy_search
                or search_type == SearchType.ctc_beam_search
            ):
                char_ids = [
                    ctc_merge_duplicates_and_remove_blanks(e, blank_id)
                    for e in char_ids
                ]
            res.append(char_ids)
        return res

    def streaming(self, attn_cache_size: int) -> "ASRModelStreaming":
        return ASRModelStreaming(self, attn_cache_size)

    def streaming_recognizer(
        self,
        search_type: str,
        attn_cache_size: int,
        beam_size: int = 1,
        blank_id: int = 0,
    ) -> BaseStreamingRecognizer:
        asr_model_streaming = self.streaming(attn_cache_size)
        if search_type == SearchType.ctc_prefix_beam_search:
            recognizer = CTCPrefixBeamSearchStreamingRecognizer(
                asr_model_streaming, beam_size, blank_id)
        elif search_type == SearchType.ctc_beam_search:
            recognizer = CTCBeamSearchStreamingRecognizer(
                asr_model_streaming, beam_size, blank_id)
        elif search_type == SearchType.ctc_greedy_search:
            recognizer = CTCGreedySearchStreamingRecognizer(
                asr_model_streaming, beam_size, blank_id)
        else:
            raise ValueError("unknown search type: %s" % search_type)
        return recognizer


class ASRModelStreaming:

    def __init__(self, asr_model: ASRModel, attn_cache_size: int):
        self.asr_model = asr_model
        self.attn_cache_size = attn_cache_size

        self.encoder_streaming = self.asr_model.encoder.streaming(
            self.attn_cache_size)
        attn_mask_maker = StreamingAttentionMaskMaker(self.attn_cache_size)
        self.encoder_streaming.set_attn_mask_maker(attn_mask_maker)

    def forward_chunk(self, chunk: Tensor) -> Tensor:
        chunk = chunk.unsqueeze(0)  # (1, time, dim)
        hs = self.encoder_streaming.forward_chunk(chunk)
        ys_hat = self.asr_model.ctc_decoder(hs)
        ys_hat = ys_hat.squeeze(0)
        return ys_hat

    def clear_cache(self) -> None:
        self.encoder_streaming.clear_cache()


def create_asr_model(conf: dict, cmvn_file: str = None):
    if cmvn_file:
        mean, istd = load_cmvn(cmvn_file, True)
        global_cmvn = GlobalCMVN(
            torch.from_numpy(mean).float(),
            torch.from_numpy(istd).float())
    else:
        global_cmvn = None

    idim: int = conf.get("idim", 80)
    encoder_dim: int = conf.get("encoder_dim", 256)
    num_blocks: int = conf.get("num_blocks", 8)
    subsampling_dropout_rate: float = 0.1

    attn_num_heads: int = conf.get("attn_num_heads", 8)
    attn_d_head: int = encoder_dim // attn_num_heads
    attn_dropout_rate: float = 0.1

    feed_forward_dropout_rate: float = 0.1
    feed_forward_hidden_units: int = conf.get(
        "feed_forward_hidden_units", encoder_dim * 4)
    feed_forward_activation: nn.Module = Swish

    conv_dropout_rate: float = 0.1
    conv_kernel_size: int = 15
    conv_activation: nn.Module = Swish
    conv_causal: bool = True

    half_step_residual: bool = True

    char_size: int = int(conf["char_size"])

    history_num_frames = 8
    attn_mask_maker = CausalAttentionMaskMaker(history_num_frames)

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
        global_cmvn=global_cmvn,
    )
    encoder.set_attn_mask_maker(attn_mask_maker)
    decoder = CTCDecoder(encoder_dim, char_size)
    asr_model = ASRModel(encoder, decoder)
    return asr_model
