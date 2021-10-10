import argparse
import os

import torch
from torch.utils.data import DataLoader

import yaml

from asre2e.dataset import AudioDataset
from asre2e.dataset import AudioTransformer
from asre2e.dataset import TranscriptTransformer
from asre2e.asr_model import create_asr_model


# torch.autograd.set_detect_anomaly(True)
device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    parser = argparse.ArgumentParser(description="training your network")
    parser.add_argument("--config", required=True, help="config file")
    parser.add_argument("--train_data", required=True, help="train data file")
    parser.add_argument("--model_dir", required=True, help="save model dir")
    parser.add_argument("--checkpoint", help="checkpoint model")
    parser.add_argument("--global_cmvn", help="global cmvn file")

    args = parser.parse_args()

    conf = None
    with open(args.config) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    dataset_conf = conf["dataset"]
    dataset = AudioDataset(
        data_file=args.train_data,
        batch_size=dataset_conf["batch_size"],
        transformer=AudioTransformer(**dataset_conf["audio_transformer"]),
        target_transformer=TranscriptTransformer(),
    )
    dataloader = DataLoader(dataset, batch_size=None, shuffle=True)
    model = create_asr_model(conf["asr_model"], args.global_cmvn)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.train()

    params_require_grad = (
        "ctc_decoder.linear.weight",
        "ctc_decoder.linear.bias",
    )
    for name, param in model.named_parameters():
        if name not in params_require_grad:
            param.requires_grad = False

    learning_rate = conf["optim_conf"]["lr"]
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    epochs = conf["epochs"]
    for epoch in range(epochs):
        for i, data in enumerate(dataloader):
            _, xs, ys, xs_lengths, ys_lengths = data
            loss = model(xs, xs_lengths, ys, ys_lengths)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("epoch", epoch, "batch", i, loss.item())
        print("epoch", epoch, loss.item())
        checkpoint = os.path.join(args.model_dir, "checkpoint.%d.pt" % epoch)
        torch.save(model.state_dict(), checkpoint)


if __name__ == "__main__":
    main()
