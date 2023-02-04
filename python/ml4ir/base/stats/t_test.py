import numpy as np
import scipy
import warnings
import pandas as pd
from scipy import stats
from scipy.optimize import brenth
from pingouin import power_ttest


class StreamVariance:
    # Wrapper class to store mean, variance in batches
    count = 0
    mean = 0
    var = 0


def perform_click_rank_dist_paired_t_test(mean, variance, n):
    """
    Compute the paired t-test statistic and its p-value given mean, standard deviation and sample count
    Parameters
    ----------
    mean: float
        The mean of the rank differences for the entire dataset
    variance: float
        The variance of the rank differences for the entire dataset
    n: int
        The number of samples in the entire dataset

    Returns
    -------
    t_test_stat: float
        The t-test statistic
    pvalue: float
        The p-value of the t-test statistic
    """
    t_test_stat = mean / (np.sqrt(variance/n))
    df = n - 1
    pvalue = 1 - stats.t.cdf(np.abs(t_test_stat), df=df)
    return t_test_stat*2, pvalue*2  # multiplying by 2 for two sided t-test


def compute_stats_from_stream(diff, count, mean, m2):
    """
    Compute the running mean, variance for a stream of data.
    src: Welford's online algorithm: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

    Parameters
    ----------
    diff: List
        A batch of differences in rankings
    count: int
        Aggregates the number of samples seen so far
    mean: float
        Accumulates the mean of the rank differences so far
    m2: float
        Aggregates the squared distance from the mean

    Returns
    -------
    count: int
        The updated aggregate sample count
    mean: float
        The updated mean of the rank differences so far
    m2: float
        The updated squared distance from the mean
    """
    for i in range(len(diff)):
        count += 1
        delta = diff[i] - mean
        mean += delta / count
        delta2 = diff[i] - mean
        m2 += delta * delta2
    return count, mean, m2


def t_test_log_results(t_test_stat, pvalue, ttest_pvalue_threshold, logger):
    """
    performing click rank distribution t-test

    Parameters
    ----------
    t_test_stat: float
        The t-test statistic
    pvalue: float
        The p-value of the t-test statistic
    ttest_pvalue_threshold: float
        The p-value threshold
    logger: Logger
        Logger object to log t-test significance decision

    """
    logger.info(
        "Performing a paired t-test between the click rank distribution of new model and the old model:\n\tNull hypothesis: There is no difference between the two click distributions.\n\tAlternative hypothesis: There is a difference between the two click distributions")
    logger.info("t-test statistic={}, p-value={}".format(t_test_stat, pvalue))
    if pvalue < ttest_pvalue_threshold:
       logger.info(
            "With p-value threshold={} > p-value --> we reject the null hypothesis. The click rank distribution of the new model is significantly different from the old model".format(
                ttest_pvalue_threshold))
    else:
       logger.warning(
            "With p-value threshold={} < p-value --> we cannot reject the null hypothesis. The click rank distribution of the new model is not significantly different from the old model".format(
                ttest_pvalue_threshold))


def run_ttest(mean, variance, n, ttest_pvalue_threshold, logger):
    """
    Compute the paired t-test statistic and its p-value given mean, standard deviation and sample count
    Parameters
    ----------
    logger: Logger
        Logger object to log t-test significance decision
    mean: float
        The mean of the rank differences for the entire dataset
    variance: float
        The variance of the rank differences for the entire dataset
    n: int
        The number of samples in the entire dataset
    ttest_pvalue_threshold: float
        P-value threshold for student t-test

    Returns
    -------
    t_test_metrics_dict: Dictionary
        A dictionary with the t-test metrics recorded.
    """
    t_test_stat, pvalue = perform_click_rank_dist_paired_t_test(mean, variance, n)
    t_test_log_results(t_test_stat, pvalue, ttest_pvalue_threshold, logger)

    # Log t-test statistic and p-value
    t_test_metrics_dict = {'Rank distribution t-test statistic': t_test_stat,
                           'Rank distribution t-test pvalue': pvalue,
                           'Rank distribution t-test difference is statistically significant': pvalue < ttest_pvalue_threshold}
    return t_test_metrics_dict


def statistical_analysis_preprocessing(clicked_records, group_metric_running_variance_params, group_key, variance_list):
    """
    This function computes the mean, variance of the provided metrics for the current batch of prediction.
    ----------
    clicked_records: dataframe
            A batch of the model's prediction

    group_metric_running_variance_params: dict
        A dictionary containing intermediate mean, variance and sample size for each org for each metric

     group_key: list
        The list of keys used to aggregate the metrics

    variance_list: list
            Lists of metrics for variance computation in batches
    """

    # computing batch-wise mean and variance per metric
    metric_stat_df_list = []
    clicked_records.set_index(group_key)
    grouped = clicked_records.groupby(group_key)
    for metric in variance_list:
        grouped_agg = grouped.apply(lambda x: x[metric].mean())
        mean_df = pd.DataFrame(
            {str(group_key): grouped_agg.keys().values, str([metric, "mean"]): grouped_agg.values})

        grouped_agg = grouped.apply(lambda x: x[metric].var())
        var_df = pd.DataFrame({str(group_key): grouped_agg.keys().values, str([metric, "var"]): grouped_agg.values})

        grouped_agg = grouped.apply(lambda x: x[metric].count())
        count_df = pd.DataFrame(
            {str(group_key): grouped_agg.keys().values, str([metric, "count"]): grouped_agg.values})

        concat_df = pd.concat([mean_df, var_df, count_df], axis=1)
        metric_stat_df_list.append(concat_df)

    running_stats_df = pd.concat(metric_stat_df_list, axis=1)
    running_stats_df = running_stats_df.loc[:, ~running_stats_df.columns.duplicated()]
    # Accumulating batch-wise mean and variances
    compute_groupwise_running_variance_for_metrics(variance_list, group_metric_running_variance_params, running_stats_df, group_key)


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


def compute_required_sample_size(mean1, mean2, var1, var2, statistical_power, pvalue):
    """
    Computes the required sample size for a statistically significant change given the means and variances
    of the metrics.
    ----------
    mean1: float
        The mean of the sample 1 (E.g. baseline MRR)
    mean2: float
        The mean of the sample 2 (E.g. new MRR)
    var1: float
        The variance of the sample 1 (E.g. baseline MRR)
    var2: float
        The variance of the sample 2 (E.g. new MRR)
    statistical_power: float
        Required statistical power
    pvalue: float
        Required pvalue

    Returns
    -------
    req_sample_sz: float
        The required sample for statistically significant change
    """
    n = None
    typ = "paired"
    alternative = "two-sided"
    try:
        # compute the effect size (d)
        denominator = np.sqrt((float(var1) + float(var2)) / 2)
        if denominator == 0:
            return np.inf
        d = np.abs(float(mean1) - float(mean2)) / denominator
        req_sample_sz = power_ttest(d, n, statistical_power, pvalue, contrast=typ, alternative=alternative)
        return req_sample_sz
    except:
        return np.inf


def run_power_analysis(metric_list, group_key, group_metric_running_variance_params, statistical_power, pvalue):
    """
    Using the input's stats (mean, variance and sample size) this function computes if the metric change is
    statistical significant using the predefined statistical power and p-value
    ----------
    metric_list: list
        List of all metrics to be used in the power analysis

    group_key: list
        The list of keys used to aggregate the metrics

    group_metric_running_variance_params: dict
        A dictionary containing mean, variance and sample size for each org for each metric

    statistical_power: float
        Required statistical power

    pvalue: float
        Required pvalue

    Returns
    -------
    group_metrics_stat_sig: Pandas dataframe
        A dataframe listing each org and for each metric whether the change is statistically significant
    """
    # 00D5f000004anzC, 00DE0000000ZFsW, 00DG0000000CLUj
    group_metrics_stat_sig = []
    for group in group_metric_running_variance_params:
        group_entry = {}
        for metric in metric_list:
            new_metric = "new_" + metric
            old_metric = "old_" + metric
            sv_old = group_metric_running_variance_params[group][old_metric]
            sv_new = group_metric_running_variance_params[group][new_metric]
            req_sample_size = compute_required_sample_size(sv_old.mean, sv_new.mean, sv_old.var, sv_new.var,
                                                           statistical_power, pvalue)
            if sv_new.count >= req_sample_size:
                is_stat_sig = True
            else:
                is_stat_sig = False
            group_entry[str(group_key)] = group
            group_entry["is_"+metric+"_lift_stat_sig"] = is_stat_sig
        group_metrics_stat_sig.append(group_entry)
    return pd.DataFrame(group_metrics_stat_sig)