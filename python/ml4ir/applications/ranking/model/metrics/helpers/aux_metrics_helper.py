from typing import List

import numpy as np
import pandas as pd
from scipy.stats import rankdata

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
    return np.sum(
        [((np.power(2, relevance_grades[i]) - 1.) / np.log2(i + 1 + 1)) for i in range(len(relevance_grades))])


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


def compute_all_failure(aux_label_values: pd.Series,
                        ranks: pd.Series,
                        click_rank: int):
    """
    Computes the all failure on the aux label for given query

    Parameters
    ----------
    aux_label_values: pd.Series
        Series object containing the aux label values for a given query
    ranks: pd.Series
        Series object containing the ranks corresponding to the aux label values
    click_rank: int
        Rank of the clicked record

    Returns
    -------
    float
        All Failure value computed for the query
    """

    all_failure = 0.

    try:
        click_aux_label_value = aux_label_values[ranks == click_rank].values[0]
        pre_click_aux_label_values = aux_label_values[ranks < click_rank]

        if pre_click_aux_label_values.size > 0:
            # Query failure only if failure on all records
            all_failure = (
                1
                if (pre_click_aux_label_values < click_aux_label_value).all()
                else 0
            )

    except IndexError:
        # Ignore queries with missing or invalid click labels
        pass

    return all_failure


def compute_intrinsic_failure(aux_label_values: pd.Series,
                              ranks: pd.Series,
                              click_rank: int):
    """
    Computes the intrinsic failure on the aux label for given query

    Parameters
    ----------
    aux_label_values: pd.Series
        Series object containing the aux label values for a given query
    ranks: pd.Series
        Series object containing the ranks corresponding to the aux label values
    click_rank: int
        Rank of the clicked record

    Returns
    -------
    float
        Intrinsic Failure value computed for the query
    """
    # We need to have at least one relevant document.
    # If not, any ordering is considered ideal
    intrinsic_failure = 0.

    if aux_label_values.sum() > 0:
        # Pass the relevance grades ordered by the ranking
        intrinsic_failure = 1. - compute_ndcg(aux_label_values.values[np.argsort(ranks.values)])

    return intrinsic_failure


def compute_rank_match_failure(aux_label_values: pd.Series,
                               ranks: pd.Series,
                               click_rank: int):
    """
    Computes the rank match failure for a given query

    Parameters
    ----------
    aux_label_values: pd.Series
        Series object containing the aux label values for a given query
    ranks: pd.Series
        Series object containing the ranks corresponding to the aux label values
    click_rank: int
        Rank of the clicked record

    Returns
    -------
    float
        RankMatchFailure value computed for the query

    Notes
    -----
    Currently, only defined for queries with exactly 1 clicked record
    """
    # If click is on the first record, there is no RankMF
    if click_rank == 1:
        return 0.

    # Filter to only the records above clicked record
    aux_label_values = aux_label_values[ranks <= click_rank].values
    ranks = ranks[ranks <= click_rank].values
    assert (ranks <= click_rank).all()

    # If all records above the clicked record have some aux label value greater than 0, then there is no RankMF
    if (aux_label_values > 0.).all():
        return 0.

    # Convert aux label values to ranks
    aux_label_ranks = rankdata(aux_label_values, method="dense")

    # Convert aux ranks to relevance grades (higher is better) for use with NDCG metric
    aux_label_relevance_grades = (1. / aux_label_ranks)
    # If the aux label value is 0, then assign a relevance grade of 0 to account for variable sequence length
    aux_label_relevance_grades = aux_label_relevance_grades * (aux_label_values > 0.).astype(int)

    # If no records have a match on the title
    if aux_label_relevance_grades.sum() == 0:
        return None

    # Sort based on ranks
    aux_label_relevance_grades = aux_label_relevance_grades[ranks.argsort()]

    # Compute RankMF as 1 - NDCG
    rank_match_failure = 1 - compute_ndcg(aux_label_relevance_grades)

    return rank_match_failure


def compute_aux_metrics(aux_label_values: pd.Series,
                        ranks: pd.Series,
                        click_rank: int,
                        prefix: str = ""):
    """
    Computes the secondary ranking metrics using a aux label for a single query

    Parameters
    ----------
    aux_label_values: pd.Series
        Series object containing the aux label values for a given query
    ranks: pd.Series
        Series object containing the ranks corresponding to the aux label values
    click_rank: int
        Rank of the clicked record
    prefix: str
        Prefix attached to the metric name

    Returns
    -------
    dict
        Key value pairs of the metric names and the associated computed values
        using the aux label

    Notes
    -----
    An auxiliary label is any feature/value that serves as a proxy relevance assessment that
    the user might be interested to measure on the dataset in addition to the primary click labels.
    For example, this could be used with an exact query match feature. In that case, the metric
    sheds light on scenarios where the records with an exact match are ranked lower than those without.
    This would provide the user with complimentary information (to typical click metrics such as MRR and ACR)
    about the model to help make better trade-off decisions w.r.t. best model selection.
    """
    return {
        f"{prefix}{Metric.AUX_ALL_FAILURE}": compute_all_failure(aux_label_values, ranks, click_rank),
        f"{prefix}{Metric.AUX_INTRINSIC_FAILURE}": compute_intrinsic_failure(aux_label_values, ranks, click_rank),
        f"{prefix}{Metric.AUX_RANKMF}": compute_rank_match_failure(aux_label_values, ranks, click_rank)
    }


def compute_aux_metrics_on_query_group(query_group: pd.DataFrame,
                                       label_col: str,
                                       old_rank_col: str,
                                       new_rank_col: str,
                                       aux_label: str,
                                       group_keys: List[str] = [],
                                       ):
    """
    Compute the old and new auxiliary ranking metrics for a given
    query on a list of aux labels

    Parameters
    ----------
    query_group : `pd.DataFrame` object
        DataFrame group object for a single query to compute auxiliary metrics on
    label_col : str
        Name of the label column in the query_group
    old_rank_col : str
        Name of the column that represents the original rank of the records
    new_rank_col : str
        Name of the column that represents the newly computed rank of the records
        after reordering based on new model scores
    aux_label : str
        Features used to compute auxiliary failure metrics
    group_keys : list, optional
        List of features used to compute groupwise metrics

    Returns
    -------
    `pd.Series` object
        Series object containing the ranking metrics
        computed using the list of aux labels
        on the old and new ranks generated by the model
    Returns
    -------
    """
    aux_metrics_dict = {
        k: v[0] for k, v in query_group[group_keys].to_dict(orient="list").items()
    }
    try:
        # Compute failure stats for before and after ranking with model
        aux_metrics_dict.update(
            compute_aux_metrics(
                aux_label_values=query_group[aux_label],
                ranks=query_group[old_rank_col],
                click_rank=query_group[query_group[label_col] == 1][old_rank_col].values[0]
                if (query_group[label_col] == 1).sum() != 0
                else float("inf"),
                prefix="old_",
            )
        )
        aux_metrics_dict.update(
            compute_aux_metrics(
                aux_label_values=query_group[aux_label],
                ranks=query_group[new_rank_col],
                click_rank=query_group[query_group[label_col] == 1][new_rank_col].values[0]
                if (query_group[label_col] == 1).sum() != 0
                else float("inf"),
                prefix="new_",
            )
        )
    except IndexError:
        # Ignore queries with no/invalid click ranks
        pass

    return pd.Series(aux_metrics_dict)
