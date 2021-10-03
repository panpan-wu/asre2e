import math


def logsumexp(*log_probs: float) -> float:
    if all(log_prob == -float("inf") for log_prob in log_probs):
        return -float("inf")
    log_prob_max = max(log_probs)
    a = math.log(
        sum(
            math.exp(log_prob - log_prob_max)
            for log_prob in log_probs
        )
    )
    return log_prob_max + a
