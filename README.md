# asre2e

端到端语音识别。目前实现了 Conformer encoder 和 CTC decoder。代码的实现主要参考了 ESPnet 和 WeNet。此代码主要用来演示语音识别中常用技术的实现，包括下采样、多头注意力、因果卷积、CTC beam search、CTC prefix beam search、流式识别等技术。

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

# 安装

- 安装 Conda：请参考 https://docs.conda.io/en/latest/miniconda.html

- （可选）创建虚拟环境：

```bash
conda create -n asre2e python=3.8
conda activate asre2e
```

- 安装 PyTorch：请参考 https://pytorch.org/

- 安装 torchaudio：

```bash
conda install -c pytorch torchaudio
```

- 安装 asre2e：

```
git clone https://github.com/panpan-wu/asre2e.git
python setup.py install
```

# 训练

## 准备训练数据

- char\_map.txt

char\_map.txt 是字符和其编号的映射，格式如下：

```
<blank> 0
<unk> 1
一 2
丁 3
```

- data\_info.csv

data\_info.csv 保存了每个句子的相关信息，格式如下：

```
utterance_id,audio_file,transcript,char_ids,num_frames
BAC009S0248W0476,/home/pan/DirsShared/data/asr/aishell_test/data_aishell/wav/train/S0248/BAC009S0248W0476.wav,新京 报 记者 程子 祥 摄,1656 89 1439 3448 2993 2739 936 2699 1579,51504
```

- train.yaml

train.yaml 是训练的相关配置，格式如下：

```yaml
dataset:
    batch_size: 8
    audio_transformer:
        num_mel_bins: 80
        frame_length: 25
        frame_shift: 10

asr_model:
    idim: 80
    char_size: 4233
    encoder_dim: 256
    num_blocks: 12
    attn_num_heads: 4
    feed_forward_hidden_units: 2048

optim_conf:
    lr: 0.001

epochs: 1
```

## 运行

可以参考 egs/aishell/run.sh 和 recognize.sh。

```
egs/aishell
├── aishell.py
├── conf
│   └── train.yaml
├── data
│   ├── char_map.txt
│   ├── dev
│   │   └── data_info.csv
│   ├── test
│   │   └── data_info.csv
│   └── train
│       └── data_info.csv
├── exp
│   ├── checkpoints
│   │   └── checkpoint.0.pt
│   ├── global_cmvn.json
│   └── state_dict_from_wenet.pt
├── get_wenet_state_dict.py
├── model_params_from_wenet
│   └── 20210204_conformer_exp
│       ├── final.pt
│       ├── global_cmvn
│       ├── train.yaml
│       └── words.txt
├── recognize.sh
└── run.sh
```
