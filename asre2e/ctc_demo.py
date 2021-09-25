# 此代码只是用来演示 ctc 相关算法的计算过程，
# 其中 ctc loss 的计算参考了下列教程：
#     http://www.inf.ed.ac.uk/teaching/courses/asr/2020-21/asr16-ctc.pdf
# ctc_loss_1 相比 ctc_loss_2 有下列不同：
# 1. states 开头额外插了一个blank_id
# 2. alpha 行数多了 1，为 num_frames + 1
# 这样做可以简化 alpha 的更新逻辑。
from typing import List
from typing import Tuple


def ctc_loss_1(
    probs: List[List],
    char_ids: List,
    blank_id: int = 0,
) -> float:
    """
    Args:
        probs: 真实的概率值，不是 log 概率。shape: (num_frames, char_size),
            num_frames 是帧数，char_size 是字符表大小。
        char_ids: 字符 id，示例：[23, 47, 88]。
        blank_id: 空白符的 id，通常都是 0。
    """
    if not probs or not char_ids:
        return 0.0

    num_frames = len(probs)
    # states: [blank_id, blank_id, char_id, blank_id, char_id, ..., blank_id]
    # states 长度为 len(char_ids) * 2 + 2
    states = [blank_id]
    for char_id in char_ids:
        states.append(blank_id)
        states.append(char_id)
    states.append(blank_id)

    alpha = [[0 for _ in range(len(states))] for _ in range(num_frames + 1)]
    alpha[0][0] = 1
    for i in range(num_frames):
        for state in range(1, len(states)):
            prev = alpha[i]
            if states[state] == blank_id or states[state] == states[state - 2]:
                prob = prev[state - 1] + prev[state]
            else:
                prob = prev[state - 2] + prev[state - 1] + prev[state]
            alpha[i + 1][state] = prob * probs[i][states[state]]
    loss = (
        alpha[num_frames][len(states) - 1]
        + alpha[num_frames][len(states) - 2]
    )
    return loss


def ctc_loss_2(
    probs: List[List],
    char_ids: List,
    blank_id: int = 0,
) -> float:
    """
    Args:
        probs: 真实的概率值，不是 log 概率。shape: (num_frames, char_size),
            num_frames 是帧数，char_size 是字符表大小。
        char_ids: 字符 id，示例：[23, 47, 88]。
        blank_id: 空白符的 id，通常都是 0。
    """
    if not probs or not char_ids:
        return 0.0

    num_frames = len(probs)
    # states: [blank_id, char_id, blank_id, char_id, ..., blank_id]
    # states 长度为 len(char_ids) * 2 + 1
    states = []
    for char_id in char_ids:
        states.append(blank_id)
        states.append(char_id)
    states.append(blank_id)

    alpha = [[0 for _ in range(len(states))] for _ in range(num_frames)]
    alpha[0][0] = probs[0][states[0]]
    alpha[0][1] = probs[0][states[1]]
    for i in range(1, num_frames):
        for state in range(len(states)):
            prev = alpha[i - 1]
            # 下列逻辑可以将结果一样的进行合并
            # if state == 0:
            #     prob = prev[state]
            # elif state == 1:
            #     prob = prev[state - 1] + prev[state]
            # elif states[state] == blank_id:
            #     prob = prev[state - 1] + prev[state]
            # elif states[state] == states[state - 2]:
            #     prob = prev[state - 1] + prev[state]
            # else:
            #     prob = prev[state - 2] + prev[state - 1] + prev[state]
            if state == 0:
                prob = prev[state]
            elif (
                state == 1
                or states[state] == blank_id
                or states[state] == states[state - 1]
            ):
                prob = prev[state - 1] + prev[state]
            else:
                prob = prev[state - 2] + prev[state - 1] + prev[state]
            alpha[i][state] = prob * probs[i][states[state]]
    loss = (
        alpha[num_frames - 1][len(states) - 1]
        + alpha[num_frames - 1][len(states) - 2]
    )
    return loss


def ctc_align(
    probs: List[List],
    char_ids: List,
    blank_id: int = 0,
) -> List:
    """
    Args:
        probs: 真实的概率值，不是 log 概率。shape: (num_frames, char_size),
            num_frames 是帧数，char_size 是字符表大小。
        char_ids: 字符 id，示例：[23, 47, 88]。
        blank_id: 空白符的 id，通常都是 0。
    """
    # 字符序列比帧数还多时不可能对齐。
    if len(char_ids) > len(probs):
        raise ValueError("too few frames")
    if not probs or not char_ids:
        return []

    num_frames = len(probs)
    # states: [blank_id, blank_id, char_id, blank_id, char_id, ..., blank_id]
    # states 长度为 len(char_ids) * 2 + 2
    states = [blank_id]
    for char_id in char_ids:
        states.append(blank_id)
        states.append(char_id)
    states.append(blank_id)

    states_backtrace = [
        [0 for _ in range(len(states))]
        for _ in range(num_frames + 1)
    ]
    alpha = [
        [0.0 for _ in range(len(states))]
        for _ in range(num_frames + 1)
    ]
    alpha[0][0] = 1.0
    for frame in range(num_frames):
        for state in range(1, len(states)):
            if states[state] == blank_id or states[state] == states[state - 2]:
                froms = [
                    (alpha[frame][state], state),
                    (alpha[frame][state - 1], state - 1),
                ]
            else:
                froms = [
                    (alpha[frame][state], state),
                    (alpha[frame][state - 1], state - 1),
                    (alpha[frame][state - 2], state - 2),
                ]
            froms.sort(key=lambda e: e[0])
            max_prob, prev_state = froms[-1]
            cur_prob = max_prob * probs[frame][states[state]]
            alpha[frame + 1][state] = cur_prob
            states_backtrace[frame + 1][state] = prev_state

    prob1 = alpha[num_frames][len(states) - 1]
    prob2 = alpha[num_frames][len(states) - 2]
    max_prob = 0.0
    last_state = 0
    # 未能走到终止状态
    if prob1 == 0.0 and prob2 == 0.0:
        raise ValueError("too few frames")
    if prob1 > prob2:
        max_prob = prob1
        last_state = len(states) - 1
    else:
        max_prob = prob2
        last_state = len(states) - 2

    alignments = [states[last_state]]
    state = last_state
    for frame in range(num_frames, 1, -1):
        state = states_backtrace[frame][state]
        alignments.append(states[state])
    alignments.reverse()
    return alignments


def ctc_remove_blank(
    char_ids: List,
    blank_id: int = 0,
) -> List:
    res = []
    prev_char_id = blank_id
    for char_id in char_ids:
        if char_id != blank_id and char_id != prev_char_id:
            res.append(char_id)
        prev_char_id = char_id
    return res


def ctc_beam_search(
    probs: List[List],
    beam_size: int,
) -> List[Tuple[List, float]]:
    """
    Args:
        probs: 真实的概率值，不是 log 概率。shape: (num_frames, char_size),
            num_frames 是帧数，char_size 是字符表大小。
        beam_size: 搜索束大小。
    Returns:
        [
            ([char_id, ...], prob),
            ...
        ]
    """
    beam = [([], 1.0)]
    for frame_probs in probs:
        beam_temp = []
        for prefix, prev_prob in beam:
            for i, prob in enumerate(frame_probs):
                beam_temp.append((prefix + [i], prev_prob * prob))
        beam_temp.sort(key=lambda e: e[1], reverse=True)
        beam = beam_temp[:beam_size]
    return beam


def ctc_prefix_beam_search(
    probs: List[List],
    beam_size: int,
    blank_id: int = 0,
) -> List[Tuple[Tuple, float]]:
    """
    *a
        + 0 = *a0
        + a = *a
        + {not_a} = *a{not_a}
    *0
        + 0 = *0
        + {not_0} = *{not_0}
    """
    num_frames = len(probs)
    beam = [((blank_id,), 1.0)]
    for frame, frame_probs in enumerate(probs):
        beam_temp = {}
        for prefix, prev_prob in beam:
            for i, frame_prob in enumerate(frame_probs):
                if prefix[-1] == blank_id:
                    cur_prefix = prefix[:-1] + (i,)
                elif prefix[-1] == i:
                    cur_prefix = prefix
                else:
                    cur_prefix = prefix + (i,)
                # 如果是最后一帧数据，把最后的空白符移除。
                if frame == num_frames - 1 and cur_prefix[-1] == blank_id:
                    cur_prefix = cur_prefix[:-1]
                prob = prev_prob * frame_prob
                beam_temp[cur_prefix] = beam_temp.get(cur_prefix, 0.0) + prob
        beam_temp = sorted(
            beam_temp.items(),
            key=lambda e: e[1],
            reverse=True,
        )
        beam = beam_temp[:beam_size]
    return beam
