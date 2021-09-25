import torch
from torch import nn


def print_params_not_finite(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if param.requires_grad:
            if not torch.all(torch.isfinite(param.data)):
                print(name, "data", param.data)
            if param.grad is not None:
                if not torch.all(torch.isfinite(param.grad)):
                    print(name, "grad", param.grad)
