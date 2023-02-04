from typing import List

import numpy as np
import pandas as pd

from ml4ir.applications.ranking.model.metrics.helpers.metric_key import Metric


def compute_dcg(relevance_grades: List[float]):
    """
    Compute the discounted cumulative gains on a list of relevance grades

    Parameters
    ----------
    relevance_grades: list of float
        Relevance grades to be used to compute DCG metric
        The rank is the position in the list

    Returns
    -------
    float
        Computed DCG for the ranked list of relevance grades

    Notes
    -----
    Reference -> https://en.wikipedia.org/wiki/Discounted_cumulative_gain
    """
    return np.sum([((np.power(2, relevance_grades[i]) - 1.) / np.log2(i + 1 + 1)) for i in range(len(relevance_grades))])


def compute_ndcg(relevance_grades: List[float]):
    """
    Compute the normalized discounted cumulative gains on a list of relevance grades

    Parameters
    ----------
    relevance_grades: list of float
        Relevance grades to be used to compute NDCG metric
        The rank is the position in the list

    Returns
    -------
    float
        Computed NDCG for the ranked list of relevance grades

    Notes
    -----
    Reference -> https://en.wikipedia.org/wiki/Discounted_cumulative_gain
    """
    return compute_dcg(relevance_grades) / compute_dcg(sorted(relevance_grades, reverse=True))
