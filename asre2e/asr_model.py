import copy
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

    def __init__(
        self,
        asr_model: nn.Module,
        beam_size: int = 1,
        blank_id: int = 0,
    ):
        self.asr_model = asr_model
        self.beam_size = beam_size
        self.blank_id = blank_id

        self._prev_beam = None

    def forward_chunk(self, chunk: Tensor) -> List[List[int]]:
        """
        Args:
            chunk (Tensor): (time, dim)
        Returns:
            [List[int]]:
                [
                    [char_id, ...],
                    ...
                ]
        """
        # (1, time, dim)
        ys_hat = self.asr_model.forward_chunk(chunk)
        char_ids = self.beam_searcher.prefix_beam_search(ys_hat)
        return char_ids

    def forward(self, x: Tensor, chunk_size: int) -> List[List[int]]:
        """
        Args:
            x (Tensor): 一个完整的句子。shape: (time, dim)
            chunk_size (int): chunk 大小。
        """
        self.clear_cache()
        num_frames = x.size(0)
        for start in range(0, num_frames, chunk_size):
            chunk = x[start:start + chunk_size]
            char_ids = self.forward_chunk(chunk)
        self.clear_cache()
        return char_ids

    def clear_cache(self) -> None:
        self.asr_model.clear_cache()
        self._prev_beam = None


class CTCPrefixBeamSearchStreamingRecognizer(BaseStreamingRecognizer):

    def forward_chunk(self, chunk: Tensor) -> List[List[int]]:
        ys_hat = self.asr_model.forward_chunk(chunk)
        self._prev_beam: List[Tuple[List[int], Tuple[float, float]]] = (
            ctc_prefix_beam_search(
                ys_hat, self.beam_size, self._prev_beam, self.blank_id))
        char_ids = [e[0] for e in self._prev_beam]
        return char_ids


class CTCBeamSearchStreamingRecognizer(BaseStreamingRecognizer):

    def forward_chunk(self, chunk: Tensor) -> List[List[int]]:
        ys_hat = self.asr_model.forward_chunk(chunk)
        self._prev_beam: List[Tuple[List[int], float]] = (
            ctc_beam_search(
                ys_hat, self.beam_size, self._prev_beam, self.blank_id))
        char_ids = [
            ctc_merge_duplicates_and_remove_blanks(e[0], self.blank_id)
            for e in self._prev_beam
        ]
        return char_ids


class CTCGreedySearchStreamingRecognizer(BaseStreamingRecognizer):

    def forward_chunk(self, chunk: Tensor) -> List[List[int]]:
        ys_hat = self.asr_model.forward_chunk(chunk)
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
        """
        hs, xs_lengths = self.encoder(xs, xs_lengths)
        # (batch, time_subsampled, char_size)
        ys_hat = self.ctc_decoder(hs)
        ys_hat = ys_hat.transpose(0, 1)
        loss = self.ctc_loss(ys_hat, ys, xs_lengths, ys_lengths)
        return loss

    def forward_chunk(self, chunk: Tensor) -> Tensor:
        """
        Args:
            chunk (Tensor): (time, dim)
        Returns:
            Tensor: (time_subsampled, char_size)
        """
        chunk = chunk.unsqueeze(0)  # (1, time, dim)
        hs, _ = self.encoder(chunk, None)
        ys_hat = self.ctc_decoder(hs)
        ys_hat = ys_hat.squeeze(0)
        return ys_hat

    def streaming_recognizer(
        self,
        search_type: str,
        cache_size: int,
        beam_size: int = 1,
        blank_id: int = 0,
    ) -> BaseStreamingRecognizer:
        asr_model = copy.deepcopy(self)
        attn_mask_maker = StreamingAttentionMaskMaker(cache_size)
        asr_model.encoder.set_cache(cache_size)
        asr_model.encoder.set_attn_mask_maker(attn_mask_maker)
        if search_type == SearchType.ctc_prefix_beam_search:
            recognizer = CTCPrefixBeamSearchStreamingRecognizer(
                asr_model, beam_size, blank_id)
        elif search_type == SearchType.ctc_beam_search:
            recognizer = CTCBeamSearchStreamingRecognizer(
                asr_model, beam_size, blank_id)
        elif search_type == SearchType.ctc_greedy_search:
            recognizer = CTCGreedySearchStreamingRecognizer(
                asr_model, beam_size, blank_id)
        else:
            raise ValueError("unknown search type: %s" % search_type)
        return recognizer

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
            char_ids = [
                ctc_merge_duplicates_and_remove_blanks(e, blank_id)
                for e in char_ids
            ]
            res.append(char_ids)
        return res

    def clear_cache(self) -> None:
        self.encoder.clear_cache()


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
