from torch import Tensor
from torch import nn

from asre2e.activation import Swish


class FeedForwardModule(nn.Module):

    def __init__(
        self,
        idim: int,
        dropout_rate: float,
        activation: nn.Module = Swish,
        hidden_units: int = 0,
    ):
        super().__init__()
        if hidden_units == 0:
            hidden_units = 4 * idim
        self.layer = nn.Sequential(
            nn.LayerNorm(idim),
            nn.Linear(idim, hidden_units),
            activation(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_units, idim),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, xs: Tensor) -> Tensor:
        return self.layer(xs)
