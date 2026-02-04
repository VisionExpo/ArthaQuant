import numpy as np


def aggregate_sentiment(scores, timestamps, decay=0.9):
    """
    Time-decayed sentiment aggregation.
    More recent news has higher weight.
    """
    weights = np.array([decay ** i for i in range(len(scores))])
    weights = weights / weights.sum()
    return float(np.sum(weights * scores))
