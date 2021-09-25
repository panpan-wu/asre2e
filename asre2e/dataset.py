import csv
from typing import Callable
from typing import Tuple

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

import torchaudio
from torchaudio.compliance import kaldi


class AudioTransformer:

    def __init__(
        self,
        num_mel_bins: int = 80,
        frame_length: int = 25,
        frame_shift: int = 10,
        dither: float = 0.1,
        energy_floor: float = 0.0,
    ):
        self.num_mel_bins = num_mel_bins
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.dither = dither
        self.energy_floor = energy_floor

    def __call__(self, audio_file: str) -> Tensor:
        waveform, sample_rate = torchaudio.load(audio_file)
        waveform = waveform * (1 << 15)
        fbank = kaldi.fbank(
            waveform,
            num_mel_bins=self.num_mel_bins,
            frame_length=self.frame_length,
            frame_shift=self.frame_shift,
            dither=self.dither,
            energy_floor=self.energy_floor,
            sample_frequency=sample_rate)
        return fbank


class TranscriptTransformer:

    def __call__(self, char_ids: str) -> Tensor:
        char_id_list = [int(e) for e in char_ids.split(" ")]
        return torch.tensor(char_id_list, dtype=torch.int32)


class AudioDataset(Dataset):

    def __init__(
        self,
        data_file: str,
        batch_size: int,
        transformer: Callable[[str], Tensor],
        target_transformer: Callable[[str], Tensor],
    ):
        """
        Args:
            data_file (str): csv 格式的数据文件。
                格式： utterance_id,audio_file,transcript,char_ids,num_frames
            batch_size (int): batch 大小。
            transformer (Callable): 处理单个语音文件。
                参数：语音文件路径。
            target_transformer (Callable): 处理单个翻译文本。
                参数：翻译文本的字符编号串，例如："34 27 1032 46"
        """
        self._batch_size = batch_size
        self._transformer = transformer
        self._target_transformer = target_transformer
        self._data = []
        self._length = 0

        with open(data_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["num_frames"] = int(row["num_frames"])
                self._data.append(row)
        # 根据音频长度排序，这样可以把相近长度的音频分到一组，方便后续 pad
        self._data.sort(key=lambda e: e["num_frames"])

        self._length, r = divmod(len(self._data), self._batch_size)
        if r > 0:
            self._length += 1

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        xs = []
        ys = []
        start_index = idx * self._batch_size
        for item in self._data[start_index:start_index + self._batch_size]:
            feature = self._transformer(item["audio_file"])
            xs.append(feature)
            target = self._target_transformer(item["char_ids"])
            ys.append(target)
        xs_lengths = torch.tensor([len(e) for e in xs], dtype=torch.int32)
        ys_lengths = torch.tensor([len(e) for e in ys], dtype=torch.int32)
        xs = pad_sequence(xs, batch_first=True)
        ys = pad_sequence(ys, batch_first=True)
        return (xs, ys, xs_lengths, ys_lengths)

    def __len__(self) -> int:
        return self._length
