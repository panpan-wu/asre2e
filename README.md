# asre2e

端到端语音识别。目前实现了 Conformer encoder 和 CTC decoder。代码的实现主要参考了 espnet 和 wenet。此代码主要用来演示语音识别中常用技术的实现，包括：下采样、多头注意力、因果卷积、CTC beam search、CTC prefix beam search、流式识别等技术。

## 下采样

目前实现了 1/4 采样，即采样后的帧数大约是采样前的 1/4，代码见 asre2e/subsampling.py。

## 多头注意力

采用了带有相对位置编码的多头注意力机制，代码见 asre2e/attention.py。

## 因果卷积

因果卷积可以防止未来信息泄露，代码见 asre2e/convolution.py。

## CTC search

目前实现了 CTC greedy search、CTC beam search 和 CTC prefix beam search，代码见 asre2e/ctc.py。asre2e/ctc\_demo.py 用来演示 CTC 相关算法的实现，它的输入输出都是 Python 原生类型。

## 流式识别

训练时通过 Mask 来屏蔽掉不希望当前帧看到的信息，识别时通过 Cache 缓存一定量历史信息。Mask 相关代码见 asre2e/mask.py。Cache 相关代码见 subsampling.py、attention.py 和 convolution.py，为了实现流式识别，需要在多个地方进行缓存，具体来说只要依赖过去信息的地方都需要缓存。
