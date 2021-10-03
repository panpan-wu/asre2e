import torch

from asre2e.ctc import ctc_beam_search
from asre2e.ctc import ctc_greedy_search
from asre2e.ctc import ctc_merge_duplicates_and_remove_blanks
from asre2e.ctc import ctc_prefix_beam_search
from asre2e.utils import logsumexp


def _float_equal(a: float, b: float) -> bool:
    if a < b:
        a, b = b, a
    return (a - b) < 1e-10


def _gen_test_data():
    probs = [
        [0.1, 0.2, 0.7],
        [0.4, 0.5, 0.1],
        [0.3, 0.2, 0.5],
        [0.2, 0.5, 0.3],
    ]
    log_probs = torch.log(torch.tensor(probs))

    seqs = [(tuple(), 0.0)]
    for frame_idx in range(log_probs.size(0)):
        frame = log_probs[frame_idx]
        seqs_temp = []
        for prefix, prev_prob in seqs:
            for char_id in range(frame.size(0)):
                prob = frame[char_id].item()
                seqs_temp.append((prefix + (char_id,), prev_prob + prob))
        seqs = seqs_temp
    seqs.sort(key=lambda e: e[1], reverse=True)

    seqs_noblank = {}
    for seq, prob in seqs:
        key = tuple(ctc_merge_duplicates_and_remove_blanks(seq))
        seqs_noblank[key] = logsumexp(
            seqs_noblank.get(key, -float("inf")), prob)
    return (log_probs, seqs, seqs_noblank)


def test_ctc_merge_duplicates_and_remove_blanks():
    char_ids = [1, 1, 2, 2]
    assert ctc_merge_duplicates_and_remove_blanks(char_ids) == [1, 2]
    char_ids = [1, 0, 1, 1]
    assert ctc_merge_duplicates_and_remove_blanks(char_ids) == [1, 1]
    char_ids = [0, 0, 0, 0]
    assert ctc_merge_duplicates_and_remove_blanks(char_ids) == []
    char_ids = [0, 1, 0, 2]
    assert ctc_merge_duplicates_and_remove_blanks(char_ids) == [1, 2]


def test_ctc_greedy_search():
    log_probs, seqs, _ = _gen_test_data()
    res = ctc_greedy_search(log_probs)
    assert tuple(res) == seqs[0][0]


def test_ctc_beam_search():
    log_probs, seqs, _ = _gen_test_data()
    beam_size = 2
    beam = ctc_beam_search(log_probs, beam_size)
    for i in range(beam_size):
        seq, prob = beam[i]
        assert seq == seqs[i][0]
        assert _float_equal(prob, seqs[i][1])


def test_ctc_prefix_beam_search():
    log_probs, _, seqs_noblank = _gen_test_data()
    beam_size = 100
    beam = ctc_prefix_beam_search(log_probs, beam_size)
    for seq, probs in beam:
        prob = logsumexp(*probs)
        assert _float_equal(prob, seqs_noblank[seq])

    beam_size = 2
    beam = ctc_prefix_beam_search(log_probs, beam_size)
    for seq, probs in beam:
        prob = logsumexp(*probs)
        assert prob <= seqs_noblank[seq]
