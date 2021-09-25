import random

from asre2e.ctc_demo import ctc_align
from asre2e.ctc_demo import ctc_beam_search
from asre2e.ctc_demo import ctc_loss_1
from asre2e.ctc_demo import ctc_loss_2
from asre2e.ctc_demo import ctc_prefix_beam_search
from asre2e.ctc_demo import ctc_remove_blank


def _randprobs(length: int):
    sample = random.sample(range(length * 100), length)
    s = sum(sample)
    return [e / s for e in sample]


def _float_equal(a: float, b: float):
    if a < b:
        a, b = b, a
    return (a - b) < 1e-10


def test_ctc_loss():
    probs = [
        [0.1, 0.2, 0.7],
        [0.4, 0.5, 0.1],
        [0.3, 0.2, 0.5],
        [0.2, 0.5, 0.3],
    ]
    char_ids = [1, 2]
    num_frames = len(probs)
    possible_seqs = [
        [0, 0, 1, 2],
        [0, 1, 0, 2],
        [0, 1, 1, 2],
        [0, 1, 2, 0],
        [0, 1, 2, 2],
        [1, 0, 0, 2],
        [1, 0, 2, 0],
        [1, 0, 2, 2],
        [1, 1, 0, 2],
        [1, 1, 1, 2],
        [1, 1, 2, 0],
        [1, 1, 2, 2],
        [1, 2, 0, 0],
        [1, 2, 2, 0],
        [1, 2, 2, 2],
    ]
    prob = 0.0
    for seq in possible_seqs:
        seq_prob = 1.0
        for frame in range(num_frames):
            seq_prob *= probs[frame][seq[frame]]
        prob += seq_prob
    loss1 = ctc_loss_1(probs, char_ids)
    loss2 = ctc_loss_2(probs, char_ids)
    assert _float_equal(prob, loss1)
    assert _float_equal(prob, loss2)


def test_ctc_loss_equal():
    num_frames = 8
    char_size = 4
    char_ids = [2, 1]
    for _ in range(3):
        probs = [_randprobs(char_size) for _ in range(num_frames)]
        loss1 = ctc_loss_1(probs, char_ids)
        loss2 = ctc_loss_2(probs, char_ids)
        assert loss1 == loss2


def test_ctc_align():
    probs = [
        [0.1, 0.2, 0.7],
        [0.4, 0.5, 0.1],
        [0.3, 0.2, 0.5],
        [0.2, 0.5, 0.3],
    ]
    char_ids = [1, 2]
    num_frames = len(probs)
    possible_seqs = [
        [0, 0, 1, 2],
        [0, 1, 0, 2],
        [0, 1, 1, 2],
        [0, 1, 2, 0],
        [0, 1, 2, 2],
        [1, 0, 0, 2],
        [1, 0, 2, 0],
        [1, 0, 2, 2],
        [1, 1, 0, 2],
        [1, 1, 1, 2],
        [1, 1, 2, 0],
        [1, 1, 2, 2],
        [1, 2, 0, 0],
        [1, 2, 2, 0],
        [1, 2, 2, 2],
    ]
    max_seq = None
    max_prob = 0.0
    for seq in possible_seqs:
        seq_prob = 1.0
        for frame in range(num_frames):
            seq_prob *= probs[frame][seq[frame]]
        if seq_prob > max_prob:
            max_seq = seq
            max_prob = seq_prob
    alignments = ctc_align(probs, char_ids)
    assert alignments == max_seq


def test_ctc_remove_blank():
    char_ids = [1, 1, 2, 2]
    assert ctc_remove_blank(char_ids) == [1, 2]
    char_ids = [1, 0, 1, 1]
    assert ctc_remove_blank(char_ids) == [1, 1]
    char_ids = [0, 0, 0, 0]
    assert ctc_remove_blank(char_ids) == []
    char_ids = [0, 1, 0, 2]
    assert ctc_remove_blank(char_ids) == [1, 2]


def test_ctc_beam_search():
    probs = [
        [0.1, 0.2, 0.7],
        [0.4, 0.5, 0.1],
        [0.3, 0.2, 0.5],
        [0.2, 0.5, 0.3],
    ]
    beam_size = 2
    seqs = [([], 1.0)]
    for frame_probs in probs:
        seqs_temp = []
        for prefix, prev_prob in seqs:
            for i, prob in enumerate(frame_probs):
                seqs_temp.append((prefix + [i], prev_prob * prob))
        seqs = seqs_temp
    seqs.sort(key=lambda e: e[1], reverse=True)
    beam = ctc_beam_search(probs, beam_size)
    for i in range(beam_size):
        seq, prob = beam[i]
        assert seq == seqs[i][0]
        assert _float_equal(prob, seqs[i][1])


def test_ctc_prefix_beam_search():
    probs = [
        [0.1, 0.2, 0.7],
        [0.4, 0.5, 0.1],
        [0.3, 0.2, 0.5],
        [0.2, 0.5, 0.3],
    ]
    seqs = [([], 1.0)]
    for frame_probs in probs:
        seqs_temp = []
        for prefix, prev_prob in seqs:
            for i, prob in enumerate(frame_probs):
                seqs_temp.append((prefix + [i], prev_prob * prob))
        seqs = seqs_temp
    seqs_noblank = {}
    for seq, prob in seqs:
        key = tuple(ctc_remove_blank(seq))
        seqs_noblank[key] = seqs_noblank.get(key, 0.0) + prob

    beam_size = 100
    beam = ctc_prefix_beam_search(probs, beam_size)
    for seq, prob in beam:
        assert _float_equal(prob, seqs_noblank[seq])

    beam_size = 2
    beam = ctc_prefix_beam_search(probs, beam_size)
    for seq, prob in beam:
        assert prob <= seqs_noblank[seq]
