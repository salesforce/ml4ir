import pandas as pd

from typing import List


def get_grouped_stats(
    df: pd.DataFrame,
    query_key_col: str,
    label_col: str,
    old_rank_col: str,
    new_rank_col: str,
    group_keys: List[str] = [],
):
    """Compute metrics on a data batch/chunk"""

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

    return pd.DataFrame(
        {
            "query_count": query_count,
            "sum_old_rank": sum_old_rank,
            "sum_new_rank": sum_new_rank,
            "sum_old_reciprocal_rank": sum_old_reciprocal_rank,
            "sum_new_reciprocal_rank": sum_new_reciprocal_rank,
        }
    )


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
