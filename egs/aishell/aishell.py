"""AIShell 数据准备。

1. download
download_dir
    ├── data_aishell.tgz
    └── resource_aishell.tgz

2. extract
download_dir
    ├── data_aishell
    │   ├── transcript
    │   │   └── aishell_transcript_v0.8.txt
    │   └── wav
    │       ├── dev/*/*.wav
    │       ├── test/*/*.wav
    │       └── train/*/*.wav
    ├── data_aishell.tgz
    ├── resource_aishell
    │   ├── lexicon.txt
    │   └── speaker.info
    └── resource_aishell.tgz

3. prepare data
work_dir
    ├── char_map.txt
    ├── dev
    │   └── data_info.csv
    ├── test
    │   └── data_info.csv
    └── train
        └── data_info.csv
csv 文件示例：
utterance_id,audio_file,transcript,char_ids,num_frames
BAC009S0248W0476,/download_dir/data_aishell/wav/train/S0248/BAC009S0248W0476.wav,新京 报 记者 程子 祥 摄,1656 89 1439 3448 2993 2739 936 2699 1579,51504
BAC009S0248W0477,/download_dir/data_aishell/wav/train/S0248/BAC009S0248W0477.wav,今年 狂犬 病 已死 亡 八 人,102 1149 2362 2357 2518 1115 1945 80 273 95,56301
"""
import argparse
import csv
import locale
import os
from pathlib import Path

import torchaudio
from torchaudio.datasets.utils import download_url
from torchaudio.datasets.utils import extract_archive


# 如果需要按中文字符集排序，设为 cn
cn = "zh_CN.UTF-8"
us = "en_US.UTF-8"
locale.setlocale(locale.LC_COLLATE, us)

WORK_DIR = "data"
UNK_CHAR = "<unk>"


def main():
    parser = argparse.ArgumentParser(description="prepare aishell data")
    parser.add_argument("--download_dir", required=True)
    parser.add_argument(
        "--work_dir",
        default=WORK_DIR,
        help="work dir. default: data",
    )
    parser.add_argument(
        "--unk_char",
        default=UNK_CHAR,
        help="unknown char. default: <unk>",
    )
    args = parser.parse_args()
    run(args.download_dir, args.work_dir, args.unk_char)


def run(download_dir: str, work_dir: str, unk_char: str) -> None:
    for d in (download_dir, work_dir):
        if not os.path.isdir(d):
            os.makedirs(d)
    download(download_dir)
    print("download finished")
    extract(download_dir)
    print("extract finished")
    prepare_data(download_dir, work_dir, unk_char)
    print("done")


def download(download_dir: str) -> None:
    base_url = "http://www.openslr.org/resources/33"
    for filename in ["data_aishell.tgz", "resource_aishell.tgz"]:
        url = "/".join([base_url, filename])
        file_path = os.path.join(download_dir, filename)
        if os.path.exists(file_path):
            print("jump existed file:", filename)
            continue
        download_url(url, download_dir, filename)


def extract(download_dir: str) -> None:
    for filename in ["data_aishell.tgz", "resource_aishell.tgz"]:
        if os.path.isdir(os.path.join(download_dir, filename[:-4])):
            print("jump extracted file:", filename)
            continue
        print("extrating", filename)
        file_path = os.path.join(download_dir, filename)
        extract_archive(file_path)
    wav_dir = os.path.join(download_dir, "data_aishell", "wav")
    tarfiles = [f for f in os.listdir(wav_dir) if f.endswith(".tar.gz")]
    print("tar file number:", len(tarfiles))
    for i, filename in enumerate(tarfiles, start=1):
        filepath = os.path.join(wav_dir, filename)
        print("extracting:", i, filename)
        extract_archive(filepath)
        os.remove(filepath)


def prepare_data(download_dir: str, work_dir: str, unk_char: str) -> None:
    csv_header = [
        "utterance_id", "audio_file", "transcript", "char_ids", "num_frames",
    ]
    transcript_map = get_transcript_map(download_dir)
    char_map = get_char_map(download_dir, unk_char)
    unk_char_id = char_map[unk_char]
    with open(os.path.join(work_dir, "char_map.txt"), "w") as f:
        for char, char_id in sorted(char_map.items(), key=lambda e: e[1]):
            f.write("%s %s\n" % (char, char_id))

    for typ in ("train", "dev", "test"):
        dest_dir = os.path.join(work_dir, typ)
        if not os.path.isdir(dest_dir):
            os.makedirs(dest_dir)
        wav_dir = os.path.join(download_dir, "data_aishell", "wav", typ)
        with open(os.path.join(dest_dir, "data_info.csv"), "w") as f:
            writer = csv.writer(f, delimiter=",", quotechar='"')
            writer.writerow(csv_header)
            for path in Path(wav_dir).glob("*/*.wav"):
                utterance_id = os.path.basename(path)[:-4]
                if utterance_id not in transcript_map:
                    print("Warning: %s doesn't have corresponding transcript" % utterance_id)
                    continue
                transcript = transcript_map[utterance_id]
                char_ids = transcript_to_char_ids(
                    transcript, char_map, unk_char_id)
                assert char_ids
                char_ids_str = " ".join([str(e) for e in char_ids])
                num_frames = torchaudio.info(path).num_frames
                writer.writerow(
                    [utterance_id, path, transcript, char_ids_str, num_frames]
                )


def get_transcript_map(download_dir: str) -> dict:
    transcript_file = os.path.join(
        download_dir,
        "data_aishell",
        "transcript",
        "aishell_transcript_v0.8.txt",
    )
    m = {}
    with open(transcript_file) as f:
        for line in f:
            parts = line.split(" ", 1)
            utterance_id = parts[0].strip()
            transcript = parts[1].strip()
            m[utterance_id] = transcript
    return m


def get_char_map(download_dir: str, unk_char: str = "<unk>") -> dict:
    transcript_map = get_transcript_map(download_dir)
    train_wav_dir = os.path.join(download_dir, "data_aishell", "wav", "train")
    chars = set()
    for path in Path(train_wav_dir).glob("*/*.wav"):
        utterance_id = os.path.basename(path)[:-4]
        if utterance_id not in transcript_map:
            continue
        transcript = transcript_map[utterance_id]
        for char in transcript:
            char = char.strip()
            if char:
                chars.add(char)
    char_map = {}
    char_map["<blank>"] = 0
    char_map[unk_char] = 1
    for i, char in enumerate(sorted(chars, key=locale.strxfrm), start=2):
        char_map[char] = i
    char_map["<sos/eos>"] = i + 1
    return char_map


def transcript_to_char_ids(
    transcript: str,
    char_map: dict,
    unk_char_id: int,
) -> []:
    char_ids = []
    for char in transcript:
        char = char.strip()
        if not char:
            continue
        char_id = char_map.get(char, unk_char_id)
        char_ids.append(char_id)
    return char_ids


if __name__ == "__main__":
    main()
