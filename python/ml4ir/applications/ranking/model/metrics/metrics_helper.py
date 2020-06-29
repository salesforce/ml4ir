import pandas as pd

from typing import List


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
        for secondary_label in secondary_labels:
            click_secondary_label = query_group[query_group[label_col] == 1][
                secondary_label
            ].values[0]

            old_pre_click_secondary_label = query_group[
                query_group[old_rank_col] < old_click_rank
            ][secondary_label]
            new_pre_click_secondary_label = query_group[
                query_group[new_rank_col] < new_click_rank
            ][secondary_label]

            old_failure_count = 0
            old_failure_partial_count = 0
            old_failure_score = 0
            old_failure_score_normalized = 0
            if old_pre_click_secondary_label.size > 0:
                # Query failure only if failure on all records
                old_failure_count = (
                    1 if (old_pre_click_secondary_label < click_secondary_label).all() else 0
                )
                # Query failure if failure on at least one record
                old_failure_partial_count = (
                    1 if (old_pre_click_secondary_label < click_secondary_label).any() else 0
                )
                # Count of failure records
                old_failure_score = (old_pre_click_secondary_label < click_secondary_label).sum()
                # Normalizing to fraction of potential records
                old_failure_score_normalized = old_failure_score / (old_click_rank - 1)

            new_failure_count = 0
            new_failure_partial_count = 0
            new_failure_score = 0
            new_failure_score_normalized = 0
            if new_pre_click_secondary_label.size > 0:
                # Query failure only if failure on all records
                new_failure_count = (
                    1 if (new_pre_click_secondary_label < click_secondary_label).all() else 0
                )
                # Query failure if failure on at least one record
                new_failure_partial_count = (
                    1 if (new_pre_click_secondary_label < click_secondary_label).any() else 0
                )
                # Count of failure records
                new_failure_score = (new_pre_click_secondary_label < click_secondary_label).sum()
                # Normalizing to fraction of potential records
                new_failure_score_normalized = new_failure_score / (new_click_rank - 1)

            failure_metrics_dict.update(
                {
                    "{}_old_failure_count".format(secondary_label): old_failure_count,
                    "{}_new_failure_count".format(secondary_label): new_failure_count,
                    "{}_old_failure_partial_count".format(
                        secondary_label
                    ): old_failure_partial_count,
                    "{}_new_failure_partial_count".format(
                        secondary_label
                    ): new_failure_partial_count,
                    "{}_old_failure_score".format(secondary_label): old_failure_score,
                    "{}_new_failure_score".format(secondary_label): new_failure_score,
                    "{}_old_failure_score_normalized".format(
                        secondary_label
                    ): old_failure_score_normalized,
                    "{}_new_failure_score_normalized".format(
                        secondary_label
                    ): new_failure_score_normalized,
                }
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

    query_count = df_grouped["query_count"].sum()

    old_acr = df_grouped["sum_old_rank"].sum() / query_count
    new_acr = df_grouped["sum_new_rank"].sum() / query_count
    perc_improv_acr = ((old_acr - new_acr) / old_acr) * 100.0

    old_mrr = df_grouped["sum_old_reciprocal_rank"].sum() / query_count
    new_mrr = df_grouped["sum_new_reciprocal_rank"].sum() / query_count
    perc_improv_mrr = ((new_mrr - old_mrr) / old_mrr) * 100.0

    return pd.Series(
        {
            "old_ACR": old_acr,
            "new_ACR": new_acr,
            "old_MRR": old_mrr,
            "new_MRR": new_mrr,
            "perc_improv_ACR": perc_improv_acr,
            "perc_improv_MRR": perc_improv_mrr,
            "query_count": query_count,
        }
    )
