import numpy as np
import pandas as pd
from scipy import stats
import unittest
import warnings
from ml4ir.applications.ranking.t_test import perform_click_rank_dist_paired_t_test, compute_stats_from_stream, \
    compute_org_running_variance_for_metrics, StreamVariance, run_power_analysis, compute_required_sample_size

warnings.filterwarnings("ignore")

np.random.seed(123)


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

    def prepare_variance_stream_test(self, n, batch_size, metric_list, group_key, group_metric_running_variance_params, a_bucket, b_bucket):
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
            compute_org_running_variance_for_metrics(metric_list, group_metric_running_variance_params,
                                                     running_stats_df, group_key)

    def test_stream_variance_computation(self):
        n = 100
        batch_size = 10
        a_bucket = np.random.randn(n)
        b_bucket = np.random.randn(n) + 0.5
        n = len(a_bucket)
        group_metric_running_variance_params = {}
        var_metric_list = ['old_metric1', 'new_metric1', 'old_metric2', 'new_metric2']
        metric_list = ["metric1", "metric2"]
        group_key = ["organization_id"]
        self.prepare_variance_stream_test(n, batch_size, var_metric_list, group_key, group_metric_running_variance_params, a_bucket, b_bucket)
        for i in range(len(var_metric_list)):
            assert np.isclose(group_metric_running_variance_params['A'][var_metric_list[i]].mean,
                              np.mean(a_bucket * (i + 1) - i), atol=0.0001)
            assert np.isclose(group_metric_running_variance_params['A'][var_metric_list[i]].var,
                              np.var(a_bucket * (i + 1) - i, ddof=1), atol=0.0001)
            assert np.isclose(group_metric_running_variance_params['B'][var_metric_list[i]].mean,
                              np.mean(b_bucket * (i + 1) - i), atol=0.0001)
            assert np.isclose(group_metric_running_variance_params['B'][var_metric_list[i]].var,
                              np.var(b_bucket * (i + 1) - i, ddof=1), atol=0.0001)

        self.test_running_power_analysis(metric_list, group_key, group_metric_running_variance_params)

    def test_running_power_analysis(self, metric_list, group_key, group_metric_running_variance_params):
        # performing power analysis
        group_metrics_stat_sig = run_power_analysis(metric_list,
                                                    group_key,
                                                    group_metric_running_variance_params,
                                                    0.9,
                                                    0.1)
        for row in group_metrics_stat_sig.iterrows():
            for m in metric_list:
                stat_check = "is_" + m + "_lift_stat_sig"
                if m == "metric1":
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
