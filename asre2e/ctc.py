from torch import Tensor
from torch import nn


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
        return self.linear(hs).log_softmax(dim=2)
