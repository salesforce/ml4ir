import pandas as pd
import numpy as np

from typing import List, Union

# Metrics where higher value is better/desirable
POSITIVE_METRIC_SUFFIXES = [
    "MRR",
    "NDCG_mean"
]
# Metrics where lower value is better/desirable
NEGATIVE_METRIC_SUFFIXES = [
    "ACR",
    "failure_all_mean",
    "failure_any_mean",
    "failure_all_rank_mean",
    "failure_any_rank_mean",
    "failure_any_count_mean",
    "failure_any_fraction_mean"
]


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


def compute_secondary_label_metrics(
    secondary_label_values: pd.Series,
    ranks: pd.Series,
    click_rank: int,
    secondary_label: str,
    prefix: str = "",
):
    """
    Computes the secondary ranking metrics using a secondary label for a single query

    Parameters
    ----------
    secondary_label_values: pd.Series
        Series object containing the secondary label values for a given query
    ranks: pd.Series
        Series object containing the ranks corresponding to the secondary label values
    click_rank: int
        Rank of the clicked record
    secondary_label: str
        Name of the secondary label used to define metric name
    prefix: str
        Prefix attached to the metric name

    Returns
    -------
    dict
        Key value pairs of the metric names and the associated computed values
        using the secondary label

    Notes
    -----
    A secondary label is any feature/value that serves as a proxy relevance assessment that
    the user might be interested to measure on the dataset in addition to the primary click labels.
    For example, this could be used with an exact query match feature. In that case, the metric
    sheds light on scenarios where the records with an exact match are ranked lower than those without.
    This would provide the user with complimentary information (to typical click metrics such as MRR and ACR)
    about the model to help make better trade-off decisions w.r.t. best model selection.
    """
    failure_all = 0
    failure_any = 0
    failure_count = 0
    failure_fraction = 0.0
    # We need to have at least one relevant document.
    # If not, any ordering is considered ideal
    secondary_label_ndcg = 1

    try:
        click_secondary_label_value = secondary_label_values[ranks == click_rank].values[0]
        pre_click_secondary_label_values = secondary_label_values[ranks < click_rank]

        if pre_click_secondary_label_values.size > 0:
            # Query failure only if failure on all records
            failure_all = (
                1
                if (pre_click_secondary_label_values < click_secondary_label_value).all()
                else 0
            )
            # Query failure if failure on at least one record
            failure_any = (
                1
                if (pre_click_secondary_label_values < click_secondary_label_value).any()
                else 0
            )
            # Count of failure records
            failure_count = (
                pre_click_secondary_label_values < click_secondary_label_value
            ).sum()
            # Normalizing to fraction of potential records
            failure_fraction = failure_count / (click_rank - 1)

    except IndexError:
        # Ignore queries with missing or invalid click labels
        pass

    # Compute NDCG metric on the secondary label
    # NOTE: Here we are passing the relevance grades ordered by the ranking
    if secondary_label_values.sum() > 0:
        secondary_label_ndcg = compute_ndcg(
            secondary_label_values.values[np.argsort(ranks.values)]
        )

    return {
        "{}{}_NDCG".format(prefix, secondary_label): secondary_label_ndcg,
        "{}{}_failure_all".format(prefix, secondary_label): failure_all,
        "{}{}_failure_any".format(prefix, secondary_label): failure_any,
        "{}{}_failure_all_rank".format(prefix, secondary_label): click_rank
        if failure_all
        else 0,
        "{}{}_failure_any_rank".format(prefix, secondary_label): click_rank
        if failure_any
        else 0,
        "{}{}_failure_any_count".format(prefix, secondary_label): failure_count,
        "{}{}_failure_any_fraction".format(prefix, secondary_label): failure_fraction,
    }


def compute_secondary_labels_metrics_on_query_group(
    query_group: pd.DataFrame,
    label_col: str,
    old_rank_col: str,
    new_rank_col: str,
    secondary_labels: List[str],
    group_keys: List[str] = []
):
    """
    Compute the old and new secondary ranking metrics for a given
    query on a list of secondary labels

    Parameters
    ----------
    query_group : `pd.DataFrame` object
        DataFrame group object for a single query to compute secondary metrics on
    label_col : str
        Name of the label column in the query_group
    old_rank_col : str
        Name of the column that represents the original rank of the records
    new_rank_col : str
        Name of the column that represents the newly computed rank of the records
        after reordering based on new model scores
    secondary_labels : list
        List of features used to compute secondary metrics
    group_keys : list, optional
        List of features used to compute groupwise metrics

    Returns
    -------
    `pd.Series` object
        Series object containing the ranking metrics
        computed using the list of secondary labels
        on the old and new ranks generated by the model
    Returns
    -------
    """
    secondary_labels_metrics_dict = {
        k: v[0] for k, v in query_group[group_keys].to_dict(orient="list").items()
    }
    for secondary_label in secondary_labels:
        try:
            # Compute failure stats for before and after ranking with model
            secondary_labels_metrics_dict.update(
                compute_secondary_label_metrics(
                    secondary_label_values=query_group[secondary_label],
                    ranks=query_group[old_rank_col],
                    click_rank=query_group[query_group[label_col] == 1][old_rank_col].values[0]
                    if (query_group[label_col] == 1).sum() != 0
                    else float("inf"),
                    secondary_label=secondary_label,
                    prefix="old_",
                )
            )
            secondary_labels_metrics_dict.update(
                compute_secondary_label_metrics(
                    secondary_label_values=query_group[secondary_label],
                    ranks=query_group[new_rank_col],
                    click_rank=query_group[query_group[label_col] == 1][new_rank_col].values[0]
                    if (query_group[label_col] == 1).sum() != 0
                    else float("inf"),
                    secondary_label=secondary_label,
                    prefix="new_",
                )
            )
        except IndexError:
            # Ignore queries with no/invalid click ranks
            continue

    return pd.Series(secondary_labels_metrics_dict)


def get_grouped_stats(
        df: pd.DataFrame,
        query_key_col: str,
        label_col: str,
        old_rank_col: str,
        new_rank_col: str,
        group_keys: List[str] = [],
        secondary_labels: List[str] = [],
):
    """
    Compute query stats that can be used to compute ranking metrics

    Parameters
    ----------
    df : `pd.DataFrame` object
        DataFrame object to compute stats and metrics on
    query_key_col : str
        Name of the query key column
    label_col : str
        Name of the label column
    old_rank_col : str
        Name of the column that represents the original rank of the records
    new_rank_col : str
        Name of the column that represents the newly computed rank of the records
        after reordering based on new model scores
    group_keys : list, optional
        List of features used to compute groupwise metrics
    secondary_labels : list, optional
        List of features used to compute failure metrics

    Returns
    -------
    `pd.DataFrame` object
        DataFrame object containing the ranking stats and failure stats
        computed from the old and new ranks and secondary labels generated
        by the model
    """

    # Compute metrics on secondary labels
    df_secondary_labels_metrics = pd.DataFrame()
    if secondary_labels:
        df_secondary_labels_metrics = df.groupby(query_key_col).apply(
            lambda grp: compute_secondary_labels_metrics_on_query_group(
                query_group=grp,
                label_col=label_col,
                old_rank_col=old_rank_col,
                new_rank_col=new_rank_col,
                secondary_labels=secondary_labels,
                group_keys=group_keys
            ))

    # Select clicked records
    df_clicked = df[df[label_col] == 1.0]

    if group_keys:
        df_grouped_batch = df_clicked.groupby(group_keys)

        # Compute groupwise stats
        query_count = df_grouped_batch[query_key_col].nunique()
        sum_old_rank = df_grouped_batch.apply(lambda x: x[old_rank_col].sum())
        sum_new_rank = df_grouped_batch.apply(lambda x: x[new_rank_col].sum())
        sum_old_reciprocal_rank = df_grouped_batch.apply(lambda x: (1.0 / x[old_rank_col]).sum())
        sum_new_reciprocal_rank = df_grouped_batch.apply(lambda x: (1.0 / x[new_rank_col]).sum())

        # Aggregate secondary label metrics by group keys
        if secondary_labels:
            df_secondary_labels_metrics = df_secondary_labels_metrics.groupby(group_keys).sum()
    else:
        # Compute overall stats if group keys are not specified
        query_count = [df_clicked.shape[0]]
        sum_old_rank = [df_clicked[old_rank_col].sum()]
        sum_new_rank = [df_clicked[new_rank_col].sum()]
        sum_old_reciprocal_rank = [(1.0 / df_clicked[old_rank_col]).sum()]
        sum_new_reciprocal_rank = [(1.0 / df_clicked[new_rank_col]).sum()]

        # Aggregate secondary label metrics
        if secondary_labels:
            df_secondary_labels_metrics = df_secondary_labels_metrics.sum().to_frame().T

    df_label_stats = pd.DataFrame(
        {
            "query_count": query_count,
            "sum_old_rank": sum_old_rank,
            "sum_new_rank": sum_new_rank,
            "sum_old_reciprocal_rank": sum_old_reciprocal_rank,
            "sum_new_reciprocal_rank": sum_new_reciprocal_rank,
        }
    )

    df_stats = pd.concat([df_label_stats, df_secondary_labels_metrics], axis=1)

    return df_stats


def summarize_grouped_stats(df_grouped):
    """
    Summarize and compute metrics from grouped ranking data stats

    Parameters
    ----------
    df_grouped : `pd.DataFrame` object
        Grouped DataFrame object containing the query level stats
        to use for computing ranking metrics

    Returns
    -------
    `pd.DataFrame` object
        DataFrame object containing ranking metrics computed from input
        query stats dataframe
    """

    if isinstance(df_grouped, pd.Series):
        df_grouped_metrics = df_grouped.to_frame().T.sum()
    else:
        df_grouped_metrics = df_grouped.sum()

    query_count = df_grouped_metrics["query_count"]
    df_grouped_metrics = df_grouped_metrics / query_count
    df_grouped_metrics["query_count"] = query_count

    # Rename metrics appropriately
    df_grouped_metrics = df_grouped_metrics.rename(
        {c: "{}_mean".format(c) for c in df_grouped_metrics.to_dict().keys()}
    )
    df_grouped_metrics = df_grouped_metrics.rename(
        {
            "sum_old_rank_mean": "old_ACR",
            "sum_new_rank_mean": "new_ACR",
            "sum_old_reciprocal_rank_mean": "old_MRR",
            "sum_new_reciprocal_rank_mean": "new_MRR",
            "query_count_mean": "query_count",
        }
    )

    processed_metric_name_suffixes = set()
    for metric_name in df_grouped_metrics.to_dict().keys():
        metric_name_suffix = metric_name[4:]
        perc_improv_metric_name = "perc_improv_{}".format(metric_name_suffix)

        if metric_name_suffix in processed_metric_name_suffixes:
            continue

        # If higher values of the metric are better/desirable
        if metric_name_suffix.endswith(tuple(POSITIVE_METRIC_SUFFIXES)):
            df_grouped_metrics[perc_improv_metric_name] = 100. * (
                (df_grouped_metrics["new_{}".format(metric_name_suffix)] -
                 df_grouped_metrics["old_{}".format(metric_name_suffix)])
                / df_grouped_metrics["old_{}".format(metric_name_suffix)])

        # If lower values of the metric are better/desirable
        elif metric_name_suffix.endswith(tuple(NEGATIVE_METRIC_SUFFIXES)):
            df_grouped_metrics[perc_improv_metric_name] = 100. * (
                (df_grouped_metrics["old_{}".format(metric_name_suffix)] -
                 df_grouped_metrics["new_{}".format(metric_name_suffix)])
                / df_grouped_metrics["old_{}".format(metric_name_suffix)])

        processed_metric_name_suffixes.add(metric_name_suffix)

    return df_grouped_metrics
