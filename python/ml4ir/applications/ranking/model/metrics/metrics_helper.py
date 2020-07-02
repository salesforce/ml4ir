import pandas as pd

from typing import List, Union


def compute_failure_stats(
    df: pd.DataFrame,
    query_key_col: str,
    label_col: str,
    old_rank_col: str,
    new_rank_col: str,
    group_keys: List[str] = [],
    secondary_labels: List[str] = [],
):
    """Compute failure metrics using secondary labels"""

    def compute_failure_stats_group(query_group):
        old_click_rank = query_group[query_group[label_col] == 1][old_rank_col].values[0]
        new_click_rank = query_group[query_group[label_col] == 1][new_rank_col].values[0]

        failure_metrics_dict = {
            k: v[0] for k, v in query_group[group_keys].to_dict(orient="list").items()
        }

        # Define common internal function to compute failure metrics on secondary label values
        def _compute_failure_stats_internal(
            pre_click_secondary_label_values: pd.Series,
            click_secondary_label_value: Union[int, float],
            click_rank: int,
            secondary_label: str,
            prefix: str = "",
        ):
            failure_all = 0
            failure_any = 0
            failure_count = 0
            failure_fraction = 0.0
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

            return {
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

        for secondary_label in secondary_labels:
            click_secondary_label_value = query_group[query_group[label_col] == 1][
                secondary_label
            ].values[0]

            old_pre_click_secondary_label_values = query_group[
                query_group[old_rank_col] < old_click_rank
            ][secondary_label]
            new_pre_click_secondary_label_values = query_group[
                query_group[new_rank_col] < new_click_rank
            ][secondary_label]

            # Compute failure stats for before and after ranking with model
            failure_metrics_dict.update(
                _compute_failure_stats_internal(
                    pre_click_secondary_label_values=old_pre_click_secondary_label_values,
                    click_secondary_label_value=click_secondary_label_value,
                    click_rank=old_click_rank,
                    secondary_label=secondary_label,
                    prefix="old_",
                )
            )
            failure_metrics_dict.update(
                _compute_failure_stats_internal(
                    pre_click_secondary_label_values=new_pre_click_secondary_label_values,
                    click_secondary_label_value=click_secondary_label_value,
                    click_rank=new_click_rank,
                    secondary_label=secondary_label,
                    prefix="new_",
                )
            )

        return pd.Series(failure_metrics_dict)

    df_failure_stats = df.groupby(query_key_col).apply(compute_failure_stats_group)

    if group_keys:
        return df_failure_stats.groupby(group_keys).sum()
    else:
        return df_failure_stats.sum().to_frame().T


def get_grouped_stats(
    df: pd.DataFrame,
    query_key_col: str,
    label_col: str,
    old_rank_col: str,
    new_rank_col: str,
    group_keys: List[str] = [],
    secondary_labels: List[str] = [],
):
    """Compute metrics on a data batch/chunk"""

    # Compute failure metrics on secondary labels
    df_failure_stats = pd.DataFrame()
    if secondary_labels:
        df_failure_stats = compute_failure_stats(
            df=df,
            query_key_col=query_key_col,
            label_col=label_col,
            old_rank_col=old_rank_col,
            new_rank_col=new_rank_col,
            group_keys=group_keys,
            secondary_labels=secondary_labels,
        )

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
    else:
        # Compute overall stats if group keys are not specified
        query_count = [df_clicked.shape[0]]
        sum_old_rank = [df_clicked[old_rank_col].sum()]
        sum_new_rank = [df_clicked[new_rank_col].sum()]
        sum_old_reciprocal_rank = [(1.0 / df_clicked[old_rank_col]).sum()]
        sum_new_reciprocal_rank = [(1.0 / df_clicked[new_rank_col]).sum()]

    df_label_stats = pd.DataFrame(
        {
            "query_count": query_count,
            "sum_old_rank": sum_old_rank,
            "sum_new_rank": sum_new_rank,
            "sum_old_reciprocal_rank": sum_old_reciprocal_rank,
            "sum_new_reciprocal_rank": sum_new_reciprocal_rank,
        }
    )

    df_stats = pd.concat([df_label_stats, df_failure_stats], axis=1)

    return df_stats


def summarize_grouped_stats(df_grouped):
    """Summarize and compute metrics from grouped ranking data stats"""

    if isinstance(df_grouped, pd.Series):
        df_grouped_metrics = df_grouped.to_frame().T.sum()
    else:
        df_grouped_metrics = df_grouped.sum()

    query_count = df_grouped_metrics["query_count"]
    df_grouped_metrics = df_grouped_metrics / query_count
    df_grouped_metrics["query_count"] = query_count

    df_grouped_metrics = df_grouped_metrics.rename(
        {c: "mean_{}".format(c) for c in df_grouped_metrics.to_dict().keys()}
    )
    df_grouped_metrics = df_grouped_metrics.rename(
        {
            "mean_sum_old_rank": "old_ACR",
            "mean_sum_new_rank": "new_ACR",
            "mean_sum_old_reciprocal_rank": "old_MRR",
            "mean_sum_new_reciprocal_rank": "new_MRR",
            "mean_query_count": "query_count",
        }
    )

    df_grouped_metrics["perc_improv_ACR"] = (
        (df_grouped_metrics["old_ACR"] - df_grouped_metrics["new_ACR"])
        / df_grouped_metrics["old_ACR"]
    ) * 100.0
    df_grouped_metrics["perc_improv_MRR"] = (
        (df_grouped_metrics["new_MRR"] - df_grouped_metrics["old_MRR"])
        / df_grouped_metrics["old_MRR"]
    ) * 100.0
    for col in df_grouped_metrics.to_dict().keys():
        if "failure" in col:
            metric_name = col[len("mean_old_") :]
            df_grouped_metrics["perc_improv_mean_{}".format(metric_name)] = (
                df_grouped_metrics["mean_old_{}".format(metric_name)]
                - df_grouped_metrics["mean_new_{}".format(metric_name)]
            ) / df_grouped_metrics["mean_old_{}".format(metric_name)]

    return df_grouped_metrics
