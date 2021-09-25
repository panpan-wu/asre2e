import torch

from asre2e.feed_forward import FeedForwardModule


# class FeedForwardModule(nn.Module):

#     def __init__(
#         self,
#         idim: int,
#         dropout_rate: float,
#         activation: nn.Module = Swish,
#         hidden_units: int = 0,
#     ):


def test_feed_forward():
    idim = 256
    dropout_rate = 0.1
    xs = torch.randn(8, 1000, 256)
    ff = FeedForwardModule(idim, dropout_rate)
    res = ff(xs)
    assert res.size() == (8, 1000, 256)
