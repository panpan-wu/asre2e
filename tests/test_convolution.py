import torch

from asre2e.convolution import CausalConvolutionModule
from asre2e.convolution import ConvolutionModule


def test_causal_convolution():
    batch_size = 8
    time = 200
    dim = 80
    kernel_size = 15
    xs = torch.randn(batch_size, time, dim)
    xs = xs.transpose(1, 2)
    conv = CausalConvolutionModule(dim, kernel_size)
    ys = conv(xs)
    assert ys.size() == (batch_size, dim, time)

    conv.enable_cache()
    ys = conv(xs)
    assert ys.size() == (batch_size, dim, time)
    assert conv._cache.size() == (batch_size, dim, kernel_size - 1)

    time = 10
    conv.clear_cache()
    xs = torch.randn(batch_size, time, dim)
    xs = xs.transpose(1, 2)
    ys = conv(xs)
    assert ys.size() == (batch_size, dim, time)
    assert conv._cache.size() == (batch_size, dim, time)

    ys = conv(xs)
    assert ys.size() == (batch_size, dim, time)
    assert conv._cache.size() == (batch_size, dim, kernel_size - 1)


def test_convolution():
    batch_size = 8
    time = 200
    dim = 80
    kernel_size = 15
    causal = False
    xs = torch.randn(batch_size, time, dim)
    conv = ConvolutionModule(
        d_model=dim,
        dropout_rate=0.1,
        kernel_size=kernel_size,
        causal=causal,
    )
    ys = conv(xs)
    assert ys.size() == (batch_size, time, dim)

    causal = True
    conv = ConvolutionModule(
        d_model=dim,
        dropout_rate=0.1,
        kernel_size=kernel_size,
        causal=causal,
    )
    conv.enable_cache()
    ys = conv(xs)
    assert ys.size() == (batch_size, time, dim)
