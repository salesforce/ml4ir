from typing import List

import pandas as pd

from ml4ir.applications.ranking.model.metrics.helpers.metric_key import Metric
from ml4ir.applications.ranking.model.metrics.helpers.aux_metrics_helper import compute_aux_metrics_on_query_group
from ml4ir.base.stats.t_test import StreamVariance

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


def compute_groupwise_running_variance_for_metrics(metric_list, group_metric_running_variance_params, running_stats_df, group_key):
    """
    This function updates the intermediate mean and variance with each batch.
    ----------
    metric_list: list
            Lists of metrics for variance computation in batches

    group_metric_running_variance_params: dict
        A dictionary containing intermediate mean, variance and sample size for each org for each metric

    running_stats_df: pandas dataframe
        The incoming batch of mean and variance

    group_key: list
        The list of keys used to aggregate the metrics
    """
    running_stats_df = running_stats_df.fillna(0)
    for row in running_stats_df.iterrows():
        group_name = row[1][str(group_key)]
        if group_name in group_metric_running_variance_params:
            group_params = group_metric_running_variance_params[group_name]
        else:
            group_params = {}
            group_metric_running_variance_params[group_name] = group_params

        for metric in metric_list:
            new_mean = row[1][str([metric, "mean"])]
            new_var = row[1][str([metric, "var"])]
            new_count = row[1][str([metric, "count"])]

            if metric in group_params:
                sv = group_metric_running_variance_params[group_name][metric]
            else:
                sv = StreamVariance()

            # update the current mean and variance
            # https://math.stackexchange.com/questions/2971315/how-do-i-combine-standard-deviations-of-two-groups
            combined_mean = (sv.count * sv.mean + new_count * new_mean) / (sv.count + new_count)
            combine_count = sv.count + new_count
            if (sv.count + new_count - 1) != 0:
                combine_var = (((sv.count-1) * sv.var + (new_count-1) * new_var) / (sv.count + new_count - 1)) + \
                          ((sv.count * new_count * (sv.mean - new_mean)**2) / ((sv.count + new_count)*(sv.count + new_count - 1)))
            else:
                combine_var = 0

            sv.count = combine_count
            sv.mean = combined_mean
            sv.var = combine_var

            group_params[metric] = sv
            group_metric_running_variance_params[group_name] = group_params


def generate_stat_sig_based_metrics(df, metric, group_keys):
    """
    compute stats for stat sig groups

    Parameters
    ----------
    df : `pd.DataFrame` object
        prediction dataframe of aggregated results by group keys
    metric : str
        The metric used to filter the stat sig groups
    """
    stat_sig_df = df.loc[df["is_" + metric + "_lift_stat_sig"] == True]
    improved = stat_sig_df.loc[stat_sig_df["perc_improv_"+metric] >= 0]
    degraded = stat_sig_df.loc[stat_sig_df["perc_improv_"+metric] < 0]
    stat_sig_groupwise_metric_old = stat_sig_df["old_"+metric].mean()
    stat_sig_groupwise_metric_new = stat_sig_df["new_" + metric].mean()
    if metric in Metric.get_positive_metrics():
        stat_sig_groupwise_metric_improv = (stat_sig_groupwise_metric_new - stat_sig_groupwise_metric_old) /  stat_sig_groupwise_metric_old * 100
    else:
        stat_sig_groupwise_metric_improv = (stat_sig_groupwise_metric_old - stat_sig_groupwise_metric_new) / stat_sig_groupwise_metric_old * 100
    return stat_sig_df, list(improved[group_keys].values.squeeze()), list(improved[degraded].values.squeeze()), stat_sig_groupwise_metric_improv