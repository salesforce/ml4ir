from scipy import stats
import numpy as np


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
