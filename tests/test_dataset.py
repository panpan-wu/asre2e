from torch.utils.data import DataLoader

from asre2e.dataset import AudioDataset
from asre2e.dataset import AudioTransformer
from asre2e.dataset import TranscriptTransformer
from tests import TrainData


def test_dataset():
    transformer = AudioTransformer()
    target_transformer = TranscriptTransformer()
    batch_size = 8
    dataset = AudioDataset(
        TrainData.filepath,
        batch_size=batch_size,
        transformer=transformer,
        target_transformer=target_transformer,
    )
    num_batchs, r = divmod(TrainData.num_files, batch_size)
    if r > 0:
        num_batchs += 1
    length = len(dataset)
    assert length == num_batchs
    assert len(dataset[length - 1][0]) == r
    xs, ys, xs_lengths, ys_lengths = dataset[0]
    assert len(xs) == len(ys) == len(xs_lengths) == len(ys_lengths)

    dataloader = DataLoader(dataset, batch_size=None)
    for i, item in enumerate(dataloader):
        pass
    assert i == length - 1
