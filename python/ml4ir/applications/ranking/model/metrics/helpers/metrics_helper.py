from typing import List

import pandas as pd

from ml4ir.applications.ranking.model.metrics.helpers.metric_key import Metric
from ml4ir.applications.ranking.model.metrics.helpers.aux_metrics_helper import compute_aux_metrics_on_query_group

DELTA = 1e-20


def get_grouped_stats(
        df: pd.DataFrame,
        query_key_col: str,
        label_col: str,
        old_rank_col: str,
        new_rank_col: str,
        group_keys: List[str] = [],
        aux_label: str = None,
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
    aux_label : str, optional
        Feature used to compute auxiliary failure metrics

    Returns
    -------
    `pd.DataFrame` object
        DataFrame object containing the ranking stats and failure stats
        computed from the old and new ranks and aux labels generated
        by the model
    """
    # Filter unclicked queries
    df_clicked = df[df[label_col] == 1.0]
    df = df[df[query_key_col].isin(df_clicked[query_key_col])]

    # Compute metrics on aux labels
    df_aux_metrics = pd.DataFrame()
    if aux_label:
        df_aux_metrics = df.groupby(query_key_col).apply(
            lambda grp: compute_aux_metrics_on_query_group(
                query_group=grp,
                label_col=label_col,
                old_rank_col=old_rank_col,
                new_rank_col=new_rank_col,
                aux_label=aux_label,
                group_keys=group_keys
            ))

    if group_keys:
        df_grouped_batch = df_clicked.groupby(group_keys)

        # Compute groupwise stats
        query_count = df_grouped_batch[query_key_col].nunique()
        sum_old_rank = df_grouped_batch.apply(lambda x: x[old_rank_col].sum())
        sum_new_rank = df_grouped_batch.apply(lambda x: x[new_rank_col].sum())
        sum_old_reciprocal_rank = df_grouped_batch.apply(lambda x: (1.0 / x[old_rank_col]).sum())
        sum_new_reciprocal_rank = df_grouped_batch.apply(lambda x: (1.0 / x[new_rank_col]).sum())

        # Aggregate aux label metrics by group keys
        if aux_label:
            df_aux_metrics = df_aux_metrics.groupby(group_keys).sum()
    else:
        # Compute overall stats if group keys are not specified
        query_count = [df_clicked.shape[0]]
        sum_old_rank = [df_clicked[old_rank_col].sum()]
        sum_new_rank = [df_clicked[new_rank_col].sum()]
        sum_old_reciprocal_rank = [(1.0 / df_clicked[old_rank_col]).sum()]
        sum_new_reciprocal_rank = [(1.0 / df_clicked[new_rank_col]).sum()]

        # Aggregate aux label metrics
        if aux_label:
            df_aux_metrics = df_aux_metrics.sum().to_frame().T

    df_label_stats = pd.DataFrame(
        {
            "query_count": query_count,
            "sum_old_rank": sum_old_rank,
            "sum_new_rank": sum_new_rank,
            "sum_old_reciprocal_rank": sum_old_reciprocal_rank,
            "sum_new_reciprocal_rank": sum_new_reciprocal_rank,
        }
    )

    df_stats = pd.concat([df_label_stats, df_aux_metrics], axis=1)

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
        {
            "sum_old_rank": "old_ACR",
            "sum_new_rank": "new_ACR",
            "sum_old_reciprocal_rank": "old_MRR",
            "sum_new_reciprocal_rank": "new_MRR"
        }
    )

    processed_metric_name_suffixes = set()
    for metric_name in df_grouped_metrics.to_dict().keys():
        metric_name_suffix = metric_name[4:]
        perc_improv_metric_name = "perc_improv_{}".format(metric_name_suffix)

        if metric_name_suffix in processed_metric_name_suffixes:
            continue

        # If higher values of the metric are better/desirable
        if metric_name_suffix.endswith(tuple(Metric.get_positive_metrics())):
            df_grouped_metrics[perc_improv_metric_name] = 100. * (
                    (df_grouped_metrics["new_{}".format(metric_name_suffix)] -
                     df_grouped_metrics["old_{}".format(metric_name_suffix)])
                    / (df_grouped_metrics["old_{}".format(metric_name_suffix)]) + DELTA)

        # If lower values of the metric are better/desirable
        elif metric_name_suffix.endswith(tuple(Metric.get_negative_metrics())):
            df_grouped_metrics[perc_improv_metric_name] = 100. * (
                    (df_grouped_metrics["old_{}".format(metric_name_suffix)] -
                     df_grouped_metrics["new_{}".format(metric_name_suffix)])
                    / (df_grouped_metrics["old_{}".format(metric_name_suffix)]) + DELTA)

        processed_metric_name_suffixes.add(metric_name_suffix)

    return df_grouped_metrics
