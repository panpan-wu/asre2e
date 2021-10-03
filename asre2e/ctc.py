from collections import defaultdict
from typing import List
from typing import Tuple

from torch import Tensor
from torch import nn

from asre2e.utils import logsumexp


class CTCDecoder(nn.Module):

    def __init__(self, encoder_dim: int, char_size: int):
        """
        Args:
            encoder_dim (int): 编码层输出的值的维度。
            char_size (int): 字符表大小。
        """
        super().__init__()
        self.linear = nn.Linear(encoder_dim, char_size)

    def forward(self, hs: Tensor) -> Tensor:
        """
        Args:
            hs (Tensor): 编码层输出的值。shape: (batch, time, dim).
        Returns:
            Tensor: (batch, time, dim)
        """
        linear_hs = self.linear(hs)
        res = linear_hs.log_softmax(dim=2)
        return res


def ctc_merge_duplicates_and_remove_blanks(
    char_ids: List[int],
    blank_id: int = 0,
) -> List[int]:
    res = []
    prev_char_id = blank_id
    for char_id in char_ids:
        if char_id != blank_id and char_id != prev_char_id:
            res.append(char_id)
        prev_char_id = char_id
    return res


def ctc_greedy_search(log_probs: Tensor) -> List[int]:
    """
    Args:
        log_probs (Tensor): log 概率，shape: (time, char_size)
    Returns:
        List[int]:
            [char_id, ...]
    """
    # (time, 1)
    _, topk_idx = log_probs.topk(1, dim=1)
    # (time,)
    topk_idx = topk_idx.view(log_probs.size(0)).tolist()
    return topk_idx


def ctc_beam_search(
    log_probs: Tensor,
    beam_size: int,
    prev_beam: List[Tuple[List[int], float]] = None,
) -> List[Tuple[List[int], float]]:
    """
    Args:
        log_probs (Tensor): log 概率，shape: (time, char_size)
        beam_size (int): 搜索束大小。
    Returns:
        List[Tuple[List[int], float]]:
            [
                ((char_id, ...), log_prob),
                ...
            ]
    """
    if beam_size <= 0:
        beam_size = 1
    # [((prefix_id, ...), (prob_wb, prob_nb))]
    if prev_beam is None:
        beam = [(tuple(), 0.0)]
    else:
        beam = [(tuple(k), v) for k, v in prev_beam]
    num_frames = log_probs.size(0)
    for frame_idx in range(num_frames):
        beam_temp = {}
        frame = log_probs[frame_idx]
        _, topk_idx = frame.topk(beam_size)
        for prev_prefix, prev_prob in beam:
            for char_id in topk_idx:
                char_id = char_id.item()
                cur_prefix = prev_prefix + (char_id,)
                cur_prob = prev_prob + frame[char_id].item()
                beam_temp[cur_prefix] = cur_prob
        beam_temp = sorted(
            beam_temp.items(),
            key=lambda e: e[1],
            reverse=True,
        )
        beam = beam_temp[:beam_size]
    return beam


def ctc_prefix_beam_search(
    log_probs: Tensor,
    beam_size: int,
    prev_beam: List[Tuple[List[int], Tuple[float, float]]] = None,
    blank_id: int = 0,
) -> List[Tuple[List[int], Tuple[float, float]]]:
    """
    *a
        + 0 = *a0
        + a = *a
        + {not_a} = *a{not_a}
    *0
        + 0 = *0
        + {not_0} = *{not_0}

    probability_with_blank: prob_wb
    probability_no_blank: prob_nb

    Args:
        log_probs (Tensor): log 概率，shape: (time, char_size)
        beam_size (int): 搜索束大小。
    Returns:
        List[Tuple[List[int], Tuple[float, float]]]:
            [
                ((char_id, ...), (log_prob_wb, log_prob_nb)),
                ...
            ]
    """
    if beam_size <= 0:
        beam_size = 1
    # [((prefix_id, ...), (prob_wb, prob_nb))]
    if prev_beam is None:
        beam = [(tuple(), (-float("inf"), 0.0))]
    else:
        beam = [(tuple(k), v) for k, v in prev_beam]
    for frame_idx in range(log_probs.size(0)):
        beam_temp = defaultdict(lambda: (-float("inf"), -float("inf")))
        frame = log_probs[frame_idx]  # (char_size,)
        for prev_prefix, (prev_prob_wb, prev_prob_nb) in beam:
            last_char_id = prev_prefix[-1] if prev_prefix else None
            for char_id in range(frame.size(0)):
                cur_prob = frame[char_id].item()
                if char_id == blank_id:
                    # *a + 0 -> *a0
                    # *a0 + 0 -> *a0
                    cur_prefix = prev_prefix
                    cur_prob_wb, cur_prob_nb = beam_temp[cur_prefix]
                    beam_temp[cur_prefix] = (
                        logsumexp(
                            cur_prob_wb,
                            prev_prob_wb + cur_prob,
                            prev_prob_nb + cur_prob,
                        ),
                        cur_prob_nb,
                    )
                elif char_id == last_char_id:
                    # *a + a -> *a
                    cur_prefix = prev_prefix
                    cur_prob_wb, cur_prob_nb = beam_temp[cur_prefix]
                    beam_temp[cur_prefix] = (
                        cur_prob_wb,
                        logsumexp(
                            cur_prob_nb,
                            prev_prob_nb + cur_prob,
                        ),
                    )
                    # *a0 + a -> *aa
                    cur_prefix = prev_prefix + (char_id,)
                    cur_prob_wb, cur_prob_nb = beam_temp[cur_prefix]
                    beam_temp[cur_prefix] = (
                        cur_prob_wb,
                        logsumexp(
                            cur_prob_nb,
                            prev_prob_wb + cur_prob,
                        ),
                    )
                else:
                    # *a + b -> *ab
                    # *a0 + b -> *ab
                    cur_prefix = prev_prefix + (char_id,)
                    cur_prob_wb, cur_prob_nb = beam_temp[cur_prefix]
                    beam_temp[cur_prefix] = (
                        cur_prob_wb,
                        logsumexp(
                            cur_prob_nb,
                            prev_prob_wb + cur_prob,
                            prev_prob_nb + cur_prob,
                        ),
                    )
        beam_temp = sorted(
            beam_temp.items(),
            key=lambda e: logsumexp(e[1][0], e[1][1]),
            reverse=True,
        )
        beam = beam_temp[:beam_size]
    beam = [(k, v)
            for k, v in beam
            if v[0] != -float("inf") and v[1] != -float("inf")]
    return beam
