import random
import numpy as np
import pandas as pd
from scipy import stats
import unittest
import warnings
from ml4ir.applications.ranking.t_test import perform_click_rank_dist_paired_t_test, compute_stats_from_stream, \
    compute_groupwise_running_variance_for_metrics, StreamVariance, run_power_analysis, compute_required_sample_size

warnings.filterwarnings("ignore")

np.random.seed(123)
random.seed(123)


class TestStatisticalAnalysisCalculation(unittest.TestCase):
    """
    Testing the calculation of statistical analysis computations.
    """

    def t_test_calculation(self, a_bucket, b_bucket, batch_size):
        n = len(a_bucket)
        expected_t_test_stat, expected_pvalue = stats.ttest_rel(a_bucket, b_bucket)
        agg_mean, agg_count, agg_M2 = 0, 0, 0
        for bcount in range(int(n / batch_size)):
            batch_a = a_bucket[bcount * batch_size: (bcount + 1) * (batch_size)]
            batch_b = b_bucket[bcount * batch_size: (bcount + 1) * (batch_size)]
            d = batch_a - batch_b
            agg_count, agg_mean, agg_M2 = compute_stats_from_stream(d, agg_count, agg_mean, agg_M2)

        t_test_stat, pvalue = perform_click_rank_dist_paired_t_test(agg_mean, (agg_M2 / (agg_count - 1)), agg_count)
        assert np.isclose(expected_pvalue, pvalue, atol=0.0001)

    def prepare_variance_stream_test(self, n, batch_size, metric_list, group_key, group_metric_running_variance_params,
                                     a_bucket, b_bucket):
        for bcount in range(int(n / batch_size)):
            batch_a = a_bucket[bcount * batch_size: (bcount + 1) * (batch_size)]
            batch_b = b_bucket[bcount * batch_size: (bcount + 1) * (batch_size)]

            df_A = pd.DataFrame(["A"] * len(batch_a), columns=group_key)
            df_B = pd.DataFrame(["B"] * len(batch_a), columns=group_key)
            for i in range(len(metric_list)):
                df_A[metric_list[i]] = pd.DataFrame(batch_a * (i + 1) - i, columns=[metric_list[0]])
                df_B[metric_list[i]] = pd.DataFrame(batch_b * (i + 1) - i, columns=[metric_list[0]])
            concated = pd.concat([df_A, df_B], axis=0)
            metric_stat_df_list = []
            grouped = concated.groupby(group_key)
            for metric in metric_list:
                grouped_agg = grouped.apply(lambda x: x[metric].mean())
                mean_df = pd.DataFrame(
                    {str(group_key): grouped_agg.keys().values, str([metric, "mean"]): grouped_agg.values})

                grouped_agg = grouped.apply(lambda x: x[metric].var())
                var_df = pd.DataFrame(
                    {str(group_key): grouped_agg.keys().values, str([metric, "var"]): grouped_agg.values})

                grouped_agg = grouped.apply(lambda x: x[metric].count())
                count_df = pd.DataFrame(
                    {str(group_key): grouped_agg.keys().values, str([metric, "count"]): grouped_agg.values})

                concat_df = pd.concat([mean_df, var_df, count_df], axis=1)
                metric_stat_df_list.append(concat_df)

            running_stats_df = pd.concat(metric_stat_df_list, axis=1)
            running_stats_df = running_stats_df.loc[:, ~running_stats_df.columns.duplicated()]
            compute_groupwise_running_variance_for_metrics(metric_list, group_metric_running_variance_params,
                                                     running_stats_df, group_key)

    def test_stream_variance_computation(self):
        n = 100
        batch_size = 10
        a_bucket = np.array([-1.0856306, 0.99734545, 0.2829785, -1.50629471, -0.57860025, 1.65143654, -2.42667924, -0.42891263,
                    1.26593626, -0.8667404, -0.67888615, -0.09470897, 1.49138963, -0.638902, -0.44398196, -0.43435128,
                    2.20593008, 2.18678609, 1.0040539, 0.3861864, 0.73736858, 1.49073203, -0.93583387, 1.17582904,
                    -1.25388067, -0.6377515, 0.9071052, -1.4286807, -0.14006872, -0.8617549, -0.25561937, -2.79858911,
                    -1.7715331, -0.69987723, 0.92746243, -0.17363568, 0.00284592, 0.68822271, -0.87953634, 0.28362732,
                    -0.80536652, -1.72766949, -0.39089979, 0.57380586, 0.33858905, -0.01183049, 2.39236527, 0.41291216,
                    0.97873601, 2.23814334, -1.29408532, -1.03878821, 1.74371223, -0.79806274, 0.02968323, 1.06931597,
                    0.89070639, 1.75488618, 1.49564414, 1.06939267, -0.77270871, 0.79486267, 0.31427199, -1.32626546,
                    1.41729905, 0.80723653, 0.04549008, -0.23309206, -1.19830114, 0.19952407, 0.46843912, -0.83115498,
                    1.16220405, -1.09720305, -2.12310035, 1.03972709, -0.40336604, -0.12602959, -0.83751672,
                    -1.60596276, 1.25523737, -0.68886898, 1.66095249, 0.80730819, -0.31475815, -1.0859024, -0.73246199,
                    -1.21252313, 2.08711336, 0.16444123, 1.15020554, -1.26735205, 0.18103513, 1.17786194, -0.33501076,
                    1.03111446, -1.08456791, -1.36347154, 0.37940061, -0.37917643])
        b_bucket = a_bucket + 0.5
        n = len(a_bucket)
        group_metric_running_variance_params = {}
        var_metric_list = ['old_metric1', 'new_metric1', 'old_metric2', 'new_metric2']
        metric_list = ["metric1", "metric2"]
        group_key = ["organization_id"]
        self.prepare_variance_stream_test(n, batch_size, var_metric_list, group_key,
                                          group_metric_running_variance_params, a_bucket, b_bucket)
        for i in range(len(var_metric_list)):
            assert np.isclose(group_metric_running_variance_params['A'][var_metric_list[i]].mean,
                              np.mean(a_bucket * (i + 1) - i), atol=0.0001)
            assert np.isclose(group_metric_running_variance_params['A'][var_metric_list[i]].var,
                              np.var(a_bucket * (i + 1) - i, ddof=1), atol=0.0001)
            assert np.isclose(group_metric_running_variance_params['B'][var_metric_list[i]].mean,
                              np.mean(b_bucket * (i + 1) - i), atol=0.0001)
            assert np.isclose(group_metric_running_variance_params['B'][var_metric_list[i]].var,
                              np.var(b_bucket * (i + 1) - i, ddof=1), atol=0.0001)

        self.running_power_analysis_test(metric_list, group_key, group_metric_running_variance_params)

    def running_power_analysis_test(self, metric_list, group_key, group_metric_running_variance_params):
        # performing power analysis
        group_metrics_stat_sig = run_power_analysis(metric_list,
                                                    group_key,
                                                    group_metric_running_variance_params,
                                                    0.9,
                                                    0.1)
        for row in group_metrics_stat_sig.iterrows():
            for m in metric_list:
                stat_check = "is_" + m + "_lift_stat_sig"
                group_name = row[1][str(group_key)]
                if m == "metric1" and group_name == 'A':
                    assert row[1][stat_check] == True
                else:
                    assert row[1][stat_check] == False

    def test_1(self):
        n = 100
        batch_size = 10
        batch_a = np.random.randn(n) + 1
        batch_b = np.random.randn(n)
        self.t_test_calculation(batch_a, batch_b, batch_size)

    def test_2(self):
        n = 10
        batch_size = 3
        batch_a = np.random.randn(n) + 20
        batch_b = np.random.randn(n)
        self.t_test_calculation(batch_a, batch_b, batch_size)

    def test_3(self):
        n = 50
        batch_size = 5
        batch_a = np.random.randn(n)
        batch_b = np.array(batch_a)
        batch_b[0] += 1
        self.t_test_calculation(batch_a, batch_b, batch_size)

    def test_4(self):
        n = 50
        batch_size = 5
        batch_a = np.random.randn(n)
        batch_b = np.array(batch_a)
        batch_b[0] += 1
        self.t_test_calculation(batch_a, batch_b, batch_size)

    def test_5(self):
        n = 100
        batch_size = 5
        batch_a = np.random.randn(n)
        batch_b = np.random.randn(n)
        self.t_test_calculation(batch_a, batch_b, batch_size)

    def test_6(self):
        n = 100
        batch_size = 5
        batch_a = np.random.exponential(scale=0.1, size=n)
        batch_b = np.random.exponential(scale=0.5, size=n)
        self.t_test_calculation(batch_a, batch_b, batch_size)

    def test_7(self):
        n = 10
        batch_size = 2
        batch_a = np.random.exponential(scale=1.0, size=n)
        batch_b = np.random.exponential(scale=0.2, size=n)
        self.t_test_calculation(batch_a, batch_b, batch_size)

    def test_8(self):
        n = 50
        batch_size = 10
        batch_a = np.random.exponential(scale=0.1, size=n)
        batch_b = np.random.exponential(scale=0.1, size=n)
        self.t_test_calculation(batch_a, batch_b, batch_size)

    def test_9(self):
        n = 50
        batch_size = 5
        batch_a = np.random.exponential(scale=0.1, size=n)
        batch_b = batch_a + 0.001
        self.t_test_calculation(batch_a, batch_b, batch_size)


if __name__ == "__main__":
    unittest.main()
