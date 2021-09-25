import sys

import torch
from torch.utils.data import DataLoader

import yaml

from asre2e.dataset import AudioDataset
from asre2e.dataset import AudioTransformer
from asre2e.dataset import TranscriptTransformer
from asre2e.asr_model import create_asr_model


# torch.autograd.set_detect_anomaly(True)


def train():
    conf_file = sys.argv[1]
    conf = None
    with open(conf_file) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    dataset_conf = conf["dataset"]
    dataset = AudioDataset(
        data_file=dataset_conf["data_file"],
        batch_size=dataset_conf["batch_size"],
        transformer=AudioTransformer(**dataset_conf["audio_transformer"]),
        target_transformer=TranscriptTransformer(),
    )
    dataloader = DataLoader(dataset, batch_size=None, shuffle=True)
    model = create_asr_model(conf["asr_model"])
    model.train()

    learning_rate = conf["optim_conf"]["lr"]
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    epochs = conf["epochs"]
    for epoch in range(epochs):
        for i, data in enumerate(dataloader):
            xs, ys, xs_lengths, ys_lengths = data
            loss = model(xs, xs_lengths, ys, ys_lengths)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("epoch", epoch, "batch", i, loss.item())
        print("epoch", epoch, loss.item())


if __name__ == "__main__":
    train()
