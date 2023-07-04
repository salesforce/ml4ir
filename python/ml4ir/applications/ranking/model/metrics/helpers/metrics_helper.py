from typing import List

import pandas as pd
import numpy as np

from ml4ir.applications.ranking.model.metrics.helpers.metric_key import Metric
from ml4ir.applications.ranking.model.metrics.helpers.aux_metrics_helper import compute_aux_metrics_on_query_group
from ml4ir.base.stats.t_test import compute_batched_stats

DELTA = 1e-20


class RankingConstants:
    PROXY_CLICK = "proxy_click"
    NEW_RANK = "new_rank"
    NEW_MRR = "new_MRR"
    OLD_MRR = "old_MRR"
    DIFF_MRR = "diff_MRR"
    OLD_ACR = "old_ACR"
    NEW_ACR = "new_ACR"
    OLD_NDCG = "old_NDCG"
    NEW_NDCG = "new_NDCG"
    NEW_RANKING_SCORE = "ranking_score"
    TTEST_PVALUE_THRESHOLD = 0.1


def add_top_graded_relevance_record_column(df, query_key_col, ranking_score_col, label_col, new_col_name):
    """
    Adds a new column indicating the top graded relevance record per query in the DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        query_key_col(str): Name of the query key column
        ranking_score_col (str): Name of the column containing the ranking score. Used to break ties.
        label_col (str): The column name containing the relevance scores.
        new_col_name (str): The name for the new column indicating the top graded relevance.

    Returns:
        pandas.DataFrame: The input DataFrame with the new column added.
        """
    # remove queries with no relevance scores
    grouped_sum = df.groupby(query_key_col)[label_col].sum()
    queries_with_relevance_scores = grouped_sum[grouped_sum > 0]
    df_with_relevance_scores = df.loc[df[query_key_col].isin(queries_with_relevance_scores.index)]

    # Group the DataFrame by query_key_col and find the record with the highest label_col
    top_records = df_with_relevance_scores.groupby(query_key_col)[[label_col, ranking_score_col]].apply(
        lambda x: x.nlargest(1, [label_col, ranking_score_col]))

    top_indexes = np.array([idx[-1] for idx in list(top_records.transpose().to_dict().keys())])

    # Create a new column 'top_target_relevance' and initialize with 0
    df_with_relevance_scores[new_col_name] = 0.0

    # Set the 'top_target_relevance' value to 1 for the top records
    df_with_relevance_scores.loc[top_indexes, new_col_name] = 1.0

    return df_with_relevance_scores


def compute_ndcg(df, query_key_col, label_col, pred_col="ranking_score", new_col="new_NDCG"):
    """
    Computes the Normalized Discounted Cumulative Gain (NDCG) for each query in the DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        query_key_col (str): The name of the column containing the query id.
        label_col (str): The column name containing the true relevance scores.
        pred_col (str, optional): The column name containing the predicted scores. Default is "ranking_score".
        new_col (str, optional): The name for the new column containing the computed NDCG values. Default is "new_NDCG".

    Returns:
        pandas.DataFrame: The input DataFrame with the new column containing the computed NDCG values.
    """
    grouped = df.groupby(query_key_col)
    for qid, group in grouped:

        y_true = group[label_col].values
        y_pred = group[pred_col].values

        # Sort the predicted scores in descending order
        sorted_indices = np.argsort(-y_pred, axis=-1)
        sorted_labels = np.take_along_axis(y_true, sorted_indices, axis=-1)

        # Compute the Discounted Cumulative Gain (DCG)
        positions = np.arange(1, y_true.shape[-1] + 1, dtype=float)
        discounts = np.log1p(positions) / np.log1p(2.0)  # log base 2
        dcg = np.sum(sorted_labels / discounts, axis=-1)

        # Sort the true relevance scores in descending order
        true_sorted_indices = np.argsort(-y_true, axis=-1)
        true_sorted_labels = np.take_along_axis(y_true, true_sorted_indices, axis=-1)

        # Compute the Ideal Discounted Cumulative Gain (IDCG)
        idcg = np.sum(true_sorted_labels / discounts, axis=-1)

        # Compute the Normalized Discounted Cumulative Gain (NDCG)
        ndcg = dcg / idcg

        df.loc[np.array(df.loc[df[query_key_col] == qid].index), new_col] = ndcg.item()

    return df


def get_grouped_stats(
        df: pd.DataFrame,
        query_key_col: str,
        label_col: str,
        old_rank_col: str,
        new_rank_col: str,
        old_ranking_score: str = None,
        new_ranking_score: str = None,
        group_keys: List[str] = [],
        aux_label: str = None,
        power_analysis_metrics: List[str] = [],
        group_metric_running_variance_params = {},
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
    old_ranking_score : str
        Name of the column that represents the ranking score of the old model
    new_ranking_score : str
        Name of the column that represents the ranking score of the new model
    group_keys : list, optional
        List of features used to compute groupwise metrics
    aux_label : str, optional
        Feature used to compute auxiliary failure metrics
    power_analysis_metrics: list
        List of metrics require power analysis computation
    group_metric_running_variance_params: dict
        A dictionary containing intermediate mean, variance and sample size for each group for each metric

    Returns
    -------
    `pd.DataFrame` object
        DataFrame object containing the ranking stats and failure stats
        computed from the old and new ranks and aux labels generated
        by the model
    """
    if np.array([Metric.NDCG in metric for metric in power_analysis_metrics]).any():
        df = compute_ndcg(df, query_key_col, label_col, pred_col=new_ranking_score, new_col=RankingConstants.NEW_NDCG)

        if old_ranking_score in df.columns:  # if the old model ranking score is available
            df = compute_ndcg(df, query_key_col, label_col, pred_col=old_ranking_score, new_col=RankingConstants.OLD_NDCG)
        else:
            # we cannot compute old NDCG
            df[RankingConstants.OLD_NDCG] = 0.0

    # Find the records with the highest relevance grade from the label column
    df[RankingConstants.PROXY_CLICK] = df.groupby(query_key_col, group_keys=False)[label_col].apply(lambda x: x == x.max()).astype(float)
    df_clicked = df[df[RankingConstants.PROXY_CLICK] == 1.0]

    # Resolve ties by selecting the min rank from old and new ranks each
    # FIXME: The ranking_score is not used beyond this point in the code,
    #        but if it has to be, we need to do a max() for it instead of min()
    df_clicked = df_clicked.groupby(query_key_col).min().reset_index()

    # Compute metrics on aux labels
    df_aux_metrics = pd.DataFrame()
    if aux_label:
        # Filter to only queries with at least 1 click label
        df = df[df[query_key_col].isin(df_clicked[query_key_col])]

        df_aux_metrics = df.groupby(query_key_col).apply(
            lambda grp: compute_aux_metrics_on_query_group(
                query_group=grp,
                label_col=RankingConstants.PROXY_CLICK,
                old_rank_col=old_rank_col,
                new_rank_col=new_rank_col,
                aux_label=aux_label,
                group_keys=group_keys,
            ))

    # Adding ranking metrics: MRR, ACR
    df_clicked_grouped = df_clicked.groupby(query_key_col)
    df_clicked[RankingConstants.NEW_MRR] = 1.0 / df_clicked[new_rank_col]
    df_clicked[RankingConstants.OLD_MRR] = 1.0 / df_clicked[old_rank_col]
    df_clicked[RankingConstants.DIFF_MRR] = df_clicked[RankingConstants.NEW_MRR] - df_clicked[
        RankingConstants.OLD_MRR]
    df_clicked[RankingConstants.OLD_ACR] = df_clicked[old_rank_col]
    df_clicked[RankingConstants.NEW_ACR] = df_clicked[new_rank_col]

    if group_keys:
        # group df by group_keys
        df_grouped_batch = df_clicked.groupby(group_keys)

        # get the power analysis metrics column names
        primary_power_metrics = set(power_analysis_metrics) & set(
            Metric.get_metrics_with_new_old_prefix(Metric.get_all_metrics())) - set(
            Metric.get_metrics_with_new_old_prefix(Metric.get_all_aux_metrics()))

        if len(primary_power_metrics) != 0:
            # compute & accumulate batched mean, variance, count
            group_metric_running_variance_params = compute_batched_stats(df_clicked, group_metric_running_variance_params,
                                                                            group_keys, primary_power_metrics)
        # Compute groupwise stats
        query_count = df_grouped_batch[query_key_col].nunique()
        sum_old_rank = df_grouped_batch.apply(lambda x: x[old_rank_col].sum())
        sum_new_rank = df_grouped_batch.apply(lambda x: x[new_rank_col].sum())
        sum_old_reciprocal_rank = df_grouped_batch.apply(lambda x: (1.0 / x[old_rank_col]).sum())
        sum_new_reciprocal_rank = df_grouped_batch.apply(lambda x: (1.0 / x[new_rank_col]).sum())

        # Aggregate aux label metrics by group keys
        if aux_label:
            aux_power_metrics = set(power_analysis_metrics) & set(
                Metric.get_metrics_with_new_old_prefix(Metric.get_all_aux_metrics()))
            if len(aux_power_metrics) != 0:
                group_metric_running_variance_params = compute_batched_stats(df_aux_metrics,
                                                                             group_metric_running_variance_params,
                                                                             group_keys, aux_power_metrics)

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

    stats_dict = {
            "query_count": query_count,
            "sum_old_rank": sum_old_rank,
            "sum_new_rank": sum_new_rank,
            "sum_old_reciprocal_rank": sum_old_reciprocal_rank,
            "sum_new_reciprocal_rank": sum_new_reciprocal_rank,
        }

    if np.array([Metric.NDCG in m for m in power_analysis_metrics]).any():
        stats_dict[RankingConstants.NEW_NDCG] = df_grouped_batch.apply(lambda x: x[RankingConstants.NEW_NDCG].sum())
        stats_dict[RankingConstants.OLD_NDCG] = df_grouped_batch.apply(lambda x: x[RankingConstants.OLD_NDCG].sum())

    df_label_stats = pd.DataFrame(stats_dict)
    df_stats = pd.concat([df_label_stats, df_aux_metrics], axis=1)

    return df_stats, group_metric_running_variance_params, df_clicked


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


def generate_stat_sig_based_metrics(df, metric, group_keys, metrics_dict):
    """
    compute stats for stat sig groups

    Parameters
    ----------
    df : `pd.DataFrame` object
        prediction dataframe of aggregated results by group keys
    metric : str
        The metric used to filter the stat sig groups
    group_keys: List
        List of keys used to group and aggregate results
    metrics_dict: Dictionary
        Logging dictionary

    Returns
    ----------
    stat_sig_df: Dataframe
        A dataframe of only stat sig groups (Improving and degrading).
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

    improv_count = len(improved)
    degrad_count = len(degraded)
    metrics_dict["stat_sig_" + metric + "_improved_groups"] = improv_count
    metrics_dict["stat_sig_" + metric + "_degraded_groups"] = degrad_count
    metrics_dict["stat_sig_" + metric + "_group_improv_perc"] = stat_sig_groupwise_metric_improv
    metrics_dict["stat_sig_improved_" + metric + "_groups"] = improved[group_keys].values.squeeze().tolist()
    metrics_dict["stat_sig_degraded_" + metric + "_groups"] = degraded[group_keys].values.squeeze().tolist()
    return stat_sig_df


def join_stat_sig_signal(df_group_metrics, group_keys, metrics, group_metrics_stat_sig):
    """
    Parameters
    ----------
    df_group_metrics: Dataframe
        The dataframe holding the groupwise metrics
    group_keys: List
        List of keys used to group and aggregate metrics
    metrics: List
        List of metrics requiring power analysis
    group_metrics_stat_sig: Dataframe
        Dataframe containing whether the required metric is stat. sig. for each group


    Returns
    -------
    df_group_metrics: Dataframe
        Joined dataframe with the stat. sig. signals

    """
    if len(group_keys) > 0 and len(metrics) > 0:
        df_group_metrics = df_group_metrics.reset_index()
        if len(group_keys) > 1:
            df_group_metrics[str(group_keys)] = df_group_metrics[group_keys].apply(tuple, axis=1)
        else:
            df_group_metrics[str(group_keys)] = df_group_metrics[group_keys]
        df_group_metrics = pd.merge(df_group_metrics, group_metrics_stat_sig, on=str(group_keys), how='left').drop(
            columns=[str(group_keys)])
    return df_group_metrics

