import torch
from torch import Tensor


class CaucalAttentionMaskMaker:

    def __init__(self, history_num_frames: int):
        """
        Args:
            history_num_frames (int): 最多与过去多少帧进行 attention。
        """
        self.history_num_frames = history_num_frames

    def __call__(self, xs: Tensor, xs_lengths: Tensor) -> Tensor:
        """
        Args:
            xs (Tensor): (batch, time, dim)
            xs_lengths (Tensor): (batch,)
        Returns:
            Tensor: (batch, time, time)
        """
        # (time, time)
        attn_mask = make_causal_mask(
            xs.size(1), self.history_num_frames, xs.device)
        # (batch, time, time)
        attn_mask = attn_mask.unsqueeze(0).expand(
            xs.size(0), xs.size(1), xs.size(1))
        return attn_mask


def make_length_mask(lengths: Tensor) -> Tensor:
    """
    Args:
        lengths (Tensor): (batch,)
    Returns:
        Tensor: (batch, max_item_of_lengths)
    示例：
    >>> import torch
    >>> lengths = torch.tensor([6, 2, 4, 1], dtype=torch.int)
    >>> make_length_mask(lengths)
    >>> tensor([[ True,  True,  True,  True,  True,  True],
    >>>         [ True,  True, False, False, False, False],
    >>>         [ True,  True,  True,  True, False, False],
    >>>         [ True, False, False, False, False, False]])
    """
    batch_size = lengths.size(0)
    max_len = int(lengths.max().item())
    seq_range = torch.arange(
        0, max_len, dtype=torch.int64, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand < seq_length_expand
    return mask


def make_causal_mask(
    num_frames: int,
    history_num_frames: int,
    device: str,
) -> Tensor:
    """
    Args:
        num_frames (int): 总帧数。
        history_num_frames (int): 能看到的历史帧数。
        device (str): cpu or gpu
    Returns:
        Tensor: (time, time)
    示例：
    >>> make_causal_mask(8, 4, "cpu")
    >>> tensor([[ True, False, False, False, False, False, False, False],
    >>>         [ True,  True, False, False, False, False, False, False],
    >>>         [ True,  True,  True, False, False, False, False, False],
    >>>         [ True,  True,  True,  True, False, False, False, False],
    >>>         [False,  True,  True,  True,  True, False, False, False],
    >>>         [False, False,  True,  True,  True,  True, False, False],
    >>>         [False, False, False,  True,  True,  True,  True, False],
    >>>         [False, False, False, False,  True,  True,  True,  True]])
    """
    mask = torch.ones(num_frames, num_frames, device=device, dtype=torch.bool)
    mask.tril_()
    mask.triu_(-history_num_frames + 1)
    return mask
