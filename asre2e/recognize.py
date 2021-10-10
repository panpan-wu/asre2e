import argparse

import torch
from torch.utils.data import DataLoader

import yaml

from asre2e.dataset import AudioDataset
from asre2e.dataset import AudioTransformer
from asre2e.dataset import TranscriptTransformer
from asre2e.asr_model import SearchType
from asre2e.asr_model import create_asr_model


# torch.autograd.set_detect_anomaly(True)
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_char_map(char_map_file: str) -> dict:
    char_map = {}
    with open(char_map_file) as f:
        for line in f:
            char, char_id = line.split(" ", 1)
            char = char.strip()
            char_id = int(char_id.strip())
            char_map[char_id] = char
    return char_map


def main():
    parser = argparse.ArgumentParser(description="recognize test data")
    parser.add_argument("--config", required=True, help="config file")
    parser.add_argument("--test_data", required=True, help="test data file")
    parser.add_argument("--char_map", required=True, help="char map file")
    parser.add_argument("--checkpoint", required=True, help="checkpoint model")
    parser.add_argument("--global_cmvn", help="global cmvn file")
    parser.add_argument(
        "--search_type",
        default=SearchType.ctc_prefix_beam_search,
    )
    parser.add_argument("--beam_size", type=int, default=1)
    parser.add_argument("--streaming", type=bool, default=False)
    parser.add_argument("--cache_size", type=int, default=8)

    args = parser.parse_args()

    conf = None
    with open(args.config) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    char_map = load_char_map(args.char_map)

    dataset_conf = conf["dataset"]
    dataset = AudioDataset(
        data_file=args.test_data,
        batch_size=1,
        transformer=AudioTransformer(**dataset_conf["audio_transformer"]),
        target_transformer=TranscriptTransformer(),
    )
    dataloader = DataLoader(dataset, batch_size=None, shuffle=False)
    model = create_asr_model(conf["asr_model"], args.global_cmvn)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()

    if args.streaming:
        with torch.no_grad():
            recognizer = model.streaming_recognizer(
                search_type=args.search_type,
                cache_size=args.cache_size,
                beam_size=args.beam_size,
            )
            for i, data in enumerate(dataloader):
                utterance_ids, xs, _, xs_lengths, _ = data
                res = recognizer.forward(xs[0], args.cache_size)
                print(utterance_ids)
                for beam in res:
                    for item in beam:
                        chars = [char_map.get(char_id) for char_id in item]
                        print(chars)
    else:
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                utterance_ids, xs, _, xs_lengths, _ = data
                res = model.ctc_search(
                    xs, xs_lengths,
                    search_type=args.search_type,
                    beam_size=args.beam_size)
                print(utterance_ids)
                for beam in res:
                    for item in beam:
                        chars = [char_map.get(char_id) for char_id in item]
                        print(chars)


if __name__ == "__main__":
    main()
