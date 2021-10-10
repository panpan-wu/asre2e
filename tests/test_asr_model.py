import torch

from asre2e.asr_model import create_asr_model
from asre2e.asr_model import SearchType


def test():
    char_size = 1000
    model = create_asr_model({"char_size": char_size})
    batch_size = 4
    num_frames = 400
    feature_dim = 80
    beam_size = 2
    xs = torch.randn(batch_size, num_frames, feature_dim)
    xs_lengths = torch.tensor([301, 400, 397, 201], dtype=torch.int)
    search_types = [
        SearchType.ctc_prefix_beam_search,
        SearchType.ctc_beam_search,
        SearchType.ctc_greedy_search,
    ]

    for search_type in search_types:
        res = model.ctc_search(
            xs, xs_lengths, search_type=search_type, beam_size=beam_size)
        assert len(res) == batch_size
        if search_type == SearchType.ctc_greedy_search:
            for beam in res:
                assert len(beam) == 1
        else:
            for beam in res:
                assert len(beam) == beam_size

    cache_size = 16
    for search_type in search_types:
        recognizer = model.streaming_recognizer(
            search_type=search_type, cache_size=cache_size,
            beam_size=beam_size)
        res = recognizer.forward(xs, xs_lengths, cache_size)
        assert len(res) == batch_size
        if search_type == SearchType.ctc_greedy_search:
            for beam in res:
                assert len(beam) == 1
        else:
            for beam in res:
                assert len(beam) == beam_size
