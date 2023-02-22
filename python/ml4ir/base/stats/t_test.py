import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brenth
from ml4ir.applications.ranking.model.metrics.helpers import metrics_helper


class StreamVariance:
    # Wrapper class to store mean, variance and count in batches (For power analysis evaluation)
    count = 0
    mean = 0
    var = 0


def power_ttest(d=None, n=None, power=None, alpha=0.05, contrast="two-samples", alternative="two-sided"):
    """
    This is extracted from: https://github.com/raphaelvallat/pingouin/blob/master/pingouin/power.py
    Evaluate power, sample size, effect size or significance level of a one-sample T-test,
    a paired T-test or an independent two-samples T-test with equal sample sizes.
    Parameters
    ----------
    d : float
        Cohen d effect size
    n : int
        Sample size
        In case of a two-sample T-test, sample sizes are assumed to be equal.
        Otherwise, see the :py:func:`power_ttest2n` function.
    power : float
        Test power (= 1 - type II error).
    alpha : float
        Significance level (type I error probability).
        The default is 0.05.
    contrast : str
        Can be `"one-sample"`, `"two-samples"` or `"paired"`.
        Note that `"one-sample"` and `"paired"` have the same behavior.
    alternative : string
        Defines the alternative hypothesis, or tail of the test. Must be one of
        "two-sided" (default), "greater" or "less".
    Notes
    -----
    Exactly ONE of the parameters ``d``, ``n``, ``power`` and ``alpha`` must be passed as None, and
    that parameter is determined from the others.
    For a paired T-test, the sample size ``n`` corresponds to the number of pairs. For an
    independent two-sample T-test with equal sample sizes, ``n`` corresponds to the sample size of
    each group (i.e. number of observations in one group). If the sample sizes are unequal, please
    use the :py:func:`power_ttest2n` function instead.
    ``alpha`` has a default value of 0.05 so None must be explicitly passed if you want to
    compute it.
    This function is a Python adaptation of the `pwr.t.test` function implemented in the
    `pwr <https://cran.r-project.org/web/packages/pwr/pwr.pdf>`_ R package.
    Statistical power is the likelihood that a study will detect an effect when there is an effect
    there to be detected. A high statistical power means that there is a low probability of
    concluding that there is no effect when there is one. Statistical power is mainly affected by
    the effect size and the sample size.
    The first step is to use the Cohen's d to calculate the non-centrality parameter
    :math:`\\delta` and degrees of freedom :math:`v`. In case of paired groups, this is:
    .. math:: \\delta = d * \\sqrt n
    .. math:: v = n - 1
    and in case of independent groups with equal sample sizes:
    .. math:: \\delta = d * \\sqrt{\\frac{n}{2}}
    .. math:: v = (n - 1) * 2
    where :math:`d` is the Cohen d and :math:`n` the sample size.
    The critical value is then found using the percent point function of the T distribution with
    :math:`q = 1 - alpha` and :math:`v` degrees of freedom.
    Finally, the power of the test is given by the survival function of the non-central
    distribution using the previously calculated critical value, degrees of freedom and
    non-centrality parameter.
    :py:func:`scipy.optimize.brenth` is used to solve power equations for other variables (i.e.
    sample size, effect size, or significance level). If the solving fails, a nan value is
    returned.
    Results have been tested against GPower and the
    `pwr <https://cran.r-project.org/web/packages/pwr/pwr.pdf>`_ R package.
    Examples
    --------
    1. Compute power of a one-sample T-test given ``d``, ``n`` and ``alpha``
    >>> from pingouin import power_ttest
    >>> print('power: %.4f' % power_ttest(d=0.5, n=20, contrast='one-sample'))
    power: 0.5645
    2. Compute required sample size given ``d``, ``power`` and ``alpha``
    >>> print('n: %.4f' % power_ttest(d=0.5, power=0.80, alternative='greater'))
    n: 50.1508
    3. Compute achieved ``d`` given ``n``, ``power`` and ``alpha`` level
    >>> print('d: %.4f' % power_ttest(n=20, power=0.80, alpha=0.05, contrast='paired'))
    d: 0.6604
    4. Compute achieved alpha level given ``d``, ``n`` and ``power``
    >>> print('alpha: %.4f' % power_ttest(d=0.5, n=20, power=0.80, alpha=None))
    alpha: 0.4430
    5. One-sided tests
    >>> from pingouin import power_ttest
    >>> print('power: %.4f' % power_ttest(d=0.5, n=20, alternative='greater'))
    power: 0.4634
    >>> print('power: %.4f' % power_ttest(d=0.5, n=20, alternative='less'))
    power: 0.0007
    """
    # Check the number of arguments that are None
    n_none = sum([v is None for v in [d, n, power, alpha]])
    if n_none != 1:
        raise ValueError("Exactly one of n, d, power, and alpha must be None.")

    # Safety checks
    assert alternative in [
        "two-sided",
        "greater",
        "less",
    ], "Alternative must be one of 'two-sided' (default), 'greater' or 'less'."
    assert contrast.lower() in ["one-sample", "paired", "two-samples"]
    tsample = 2 if contrast.lower() == "two-samples" else 1
    tside = 2 if alternative == "two-sided" else 1
    if d is not None and tside == 2:
        d = abs(d)
    if alpha is not None:
        assert 0 < alpha <= 1
    if power is not None:
        assert 0 < power <= 1

    if alternative == "less":

        def func(d, n, power, alpha):
            dof = (n - 1) * tsample
            nc = d * np.sqrt(n / tsample)
            tcrit = stats.t.ppf(alpha / tside, dof)
            return stats.nct.cdf(tcrit, dof, nc)

    elif alternative == "two-sided":

        def func(d, n, power, alpha):
            dof = (n - 1) * tsample
            nc = d * np.sqrt(n / tsample)
            tcrit = stats.t.ppf(1 - alpha / tside, dof)
            return stats.nct.sf(tcrit, dof, nc) + stats.nct.cdf(-tcrit, dof, nc)

    else:  # Alternative = 'greater'

        def func(d, n, power, alpha):
            dof = (n - 1) * tsample
            nc = d * np.sqrt(n / tsample)
            tcrit = stats.t.ppf(1 - alpha / tside, dof)
            return stats.nct.sf(tcrit, dof, nc)

    # Evaluate missing variable
    if power is None:
        # Compute achieved power given d, n and alpha
        return func(d, n, power=None, alpha=alpha)

    elif n is None:
        # Compute required sample size given d, power and alpha

        def _eval_n(n, d, power, alpha):
            return func(d, n, power, alpha) - power

        try:
            return brenth(_eval_n, 2 + 1e-10, 1e07, args=(d, power, alpha))
        except ValueError:  # pragma: no cover
            return np.nan

    elif d is None:
        # Compute achieved d given sample size, power and alpha level
        if alternative == "two-sided":
            b0, b1 = 1e-07, 10
        elif alternative == "less":
            b0, b1 = -10, 5
        else:
            b0, b1 = -5, 10

        def _eval_d(d, n, power, alpha):
            return func(d, n, power, alpha) - power

        try:
            return brenth(_eval_d, b0, b1, args=(n, power, alpha))
        except ValueError:  # pragma: no cover
            return np.nan

    else:
        # Compute achieved alpha (significance) level given d, n and power

        def _eval_alpha(alpha, d, n, power):
            return func(d, n, power, alpha) - power

        try:
            return brenth(_eval_alpha, 1e-10, 1 - 1e-10, args=(d, n, power))
        except ValueError:  # pragma: no cover
            return np.nan


def update_running_stats_for_t_test(clicked_records, agg_count, agg_mean, agg_M2):
    """

    Parameters
    ----------
    clicked_records: Dataframe
        prediction dataframe with only clicked records
    agg_count, agg_mean, agg_M2: float
        running aggregates used to compute count, mean, variance

    Returns
    -------
    Updated version of: agg_count, agg_mean, agg_M2

    """
    diff = (clicked_records[metrics_helper.RankingConstants.DIFF_MRR]).to_list()
    agg_count, agg_mean, agg_M2 = compute_stats_from_stream(diff, agg_count, agg_mean, agg_M2)
    return agg_count, agg_mean, agg_M2


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


def compute_batched_stats(df, group_metric_running_variance_params, group_key, variance_list):
    """
    This function computes the mean, variance of the provided metrics for the current batch of prediction.
    ----------
    df: dataframe
            A batch of the model's prediction

    group_metric_running_variance_params: dict
        A dictionary containing intermediate mean, variance and sample size for each group for each metric

     group_key: list
        The list of keys used to aggregate the metrics

    variance_list: list
            Lists of metrics for variance computation in batches
    """

    # computing batch-wise mean and variance per metric
    metric_stat_df_list = []
    grouped = df.groupby(group_key)
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
    return compute_groupwise_running_variance_for_metrics(variance_list, group_metric_running_variance_params, running_stats_df, group_key)


def compute_groupwise_running_variance_for_metrics(metric_list, group_metric_running_variance_params, running_stats_df, group_key):
    """
    This function updates the intermediate mean and variance with each batch.
    ----------
    metric_list: list
            Lists of metrics for variance computation in batches

    group_metric_running_variance_params: dict
        A dictionary containing intermediate mean, variance and sample size for each group for each metric

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
            if (sv.count + new_count) == 0:
                # pypass the current batch
                combine_count = sv.count
                combined_mean = sv.mean
                combine_var = sv.var
            else:
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
    return group_metric_running_variance_params


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
        A dictionary containing mean, variance and sample size for each group for each metric

    statistical_power: float
        Required statistical power

    pvalue: float
        Required pvalue

    Returns
    -------
    group_metrics_stat_sig: Pandas dataframe
        A dataframe listing each group and for each metric whether the change is statistically significant
    """
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
