import unittest
import json

import pandas as pd
from pandas import testing as pd_testing

from ml4ir.applications.ranking.model.metrics.metrics_helper import *


class ComputeSecondaryMetricsTest(unittest.TestCase):
    """Test suite for ml4ir.applications.ranking.model.metrics.metrics_helper"""

    def test_compute_secondary_label_metrics_1(self):
        computed_metrics = compute_secondary_label_metrics(
            secondary_label_values=pd.Series([10, 10, 10, 10, 10, 10, 10, 10, 1]),
            ranks=pd.Series(list(range(1, 10))),
            click_rank=9,
            secondary_label="test_label")
        pd_testing.assert_series_equal(
            pd.Series(computed_metrics),
            pd.Series({
                "test_label_NDCG": 1.,
                "test_label_failure_all": 0,
                "test_label_failure_any": 0,
                "test_label_failure_all_rank": 0,
                "test_label_failure_any_rank": 0,
                "test_label_failure_any_count": 0,
                "test_label_failure_any_fraction": 0.0}),
            check_less_precise=True)

    def test_compute_secondary_label_metrics_2(self):
        computed_metrics = compute_secondary_label_metrics(
            secondary_label_values=pd.Series([1, 1, 1, 1, 1, 1, 1, 1, 10]),
            ranks=pd.Series(list(range(1, 10))),
            click_rank=9,
            secondary_label="test_label")
        pd_testing.assert_series_equal(
            pd.Series(computed_metrics),
            pd.Series({
                "test_label_NDCG": 0.307,
                "test_label_failure_all": 1,
                "test_label_failure_any": 1,
                "test_label_failure_all_rank": 9,
                "test_label_failure_any_rank": 9,
                "test_label_failure_any_count": 8,
                "test_label_failure_any_fraction": 1.0
            }),
            check_less_precise=True)

    def test_compute_secondary_label_metrics_3(self):
        computed_metrics = compute_secondary_label_metrics(
            secondary_label_values=pd.Series([10, 10, 10, 1]),
            ranks=pd.Series(list(range(1, 5))),
            click_rank=4,
            secondary_label="test_label")
        pd_testing.assert_series_equal(
            pd.Series(computed_metrics),
            pd.Series({
                "test_label_NDCG": 1.,
                "test_label_failure_all": 0,
                "test_label_failure_any": 0,
                "test_label_failure_all_rank": 0,
                "test_label_failure_any_rank": 0,
                "test_label_failure_any_count": 0,
                "test_label_failure_any_fraction": 0.0
            }),
            check_less_precise=True)

    def test_compute_secondary_label_metrics_4(self):
        computed_metrics = compute_secondary_label_metrics(
            secondary_label_values=pd.Series([1, 1, 1, 10]),
            ranks=pd.Series(list(range(1, 5))),
            click_rank=4,
            secondary_label="test_label")
        pd_testing.assert_series_equal(
            pd.Series(computed_metrics),
            pd.Series({
                "test_label_NDCG": 0.434,
                "test_label_failure_all": 1,
                "test_label_failure_any": 1,
                "test_label_failure_all_rank": 4,
                "test_label_failure_any_rank": 4,
                "test_label_failure_any_count": 3,
                "test_label_failure_any_fraction": 1.0
            }),
            check_less_precise=True)

    def test_compute_secondary_label_metrics_5(self):
        computed_metrics = compute_secondary_label_metrics(
            secondary_label_values=pd.Series([1, 1, 1, 5]),
            ranks=pd.Series(list(range(1, 5))),
            click_rank=4,
            secondary_label="test_label")
        pd_testing.assert_series_equal(
            pd.Series(computed_metrics),
            pd.Series({
                "test_label_NDCG": 0.514,
                "test_label_failure_all": 1,
                "test_label_failure_any": 1,
                "test_label_failure_all_rank": 4,
                "test_label_failure_any_rank": 4,
                "test_label_failure_any_count": 3,
                "test_label_failure_any_fraction": 1.0
            }),
            check_less_precise=True)

    def test_compute_secondary_label_metrics_6(self):
        computed_metrics = compute_secondary_label_metrics(
            secondary_label_values=pd.Series([5, 5, 5, 10]),
            ranks=pd.Series(list(range(1, 5))),
            click_rank=4,
            secondary_label="test_label")
        pd_testing.assert_series_equal(
            pd.Series(computed_metrics),
            pd.Series({
                "test_label_NDCG": 0.474,
                "test_label_failure_all": 1.0,
                "test_label_failure_any": 1.0,
                "test_label_failure_all_rank": 4.0,
                "test_label_failure_any_rank": 4.0,
                "test_label_failure_any_count": 3.0,
                "test_label_failure_any_fraction": 1.0
            }),
            check_less_precise=True)

    def test_compute_secondary_label_metrics_7(self):
        computed_metrics = compute_secondary_label_metrics(
            secondary_label_values=pd.Series([1, 1, 1, 1, 5, 5, 5, 5, 10]),
            ranks=pd.Series(list(range(1, 10))),
            click_rank=9,
            secondary_label="test_label")
        pd_testing.assert_series_equal(
            pd.Series(computed_metrics),
            pd.Series({
                "test_label_NDCG": 0.329,
                "test_label_failure_all": 1,
                "test_label_failure_any": 1,
                "test_label_failure_all_rank": 9,
                "test_label_failure_any_rank": 9,
                "test_label_failure_any_count": 8,
                "test_label_failure_any_fraction": 1.0
            }),
            check_less_precise=True)

    def test_compute_secondary_label_metrics_8(self):
        computed_metrics = compute_secondary_label_metrics(
            secondary_label_values=pd.Series([5, 5, 5, 5, 1, 1, 1, 1, 10]),
            ranks=pd.Series(list(range(1, 10))),
            click_rank=9,
            secondary_label="test_label")
        pd_testing.assert_series_equal(
            pd.Series(computed_metrics),
            pd.Series({
                "test_label_NDCG": 0.361,
                "test_label_failure_all": 1,
                "test_label_failure_any": 1,
                "test_label_failure_all_rank": 9,
                "test_label_failure_any_rank": 9,
                "test_label_failure_any_count": 8,
                "test_label_failure_any_fraction": 1.0
            }),
            check_less_precise=True)

    def test_compute_secondary_label_metrics_9(self):
        computed_metrics = compute_secondary_label_metrics(
            secondary_label_values=pd.Series([1, 5, 1, 5, 1, 5, 1, 5, 10]),
            ranks=pd.Series(list(range(1, 10))),
            click_rank=9,
            secondary_label="test_label")
        pd_testing.assert_series_equal(
            pd.Series(computed_metrics),
            pd.Series({
                "test_label_NDCG": 0.338,
                "test_label_failure_all": 1,
                "test_label_failure_any": 1,
                "test_label_failure_all_rank": 9,
                "test_label_failure_any_rank": 9,
                "test_label_failure_any_count": 8,
                "test_label_failure_any_fraction": 1.0
            }),
            check_less_precise=True)

    def test_compute_secondary_label_metrics_10(self):
        computed_metrics = compute_secondary_label_metrics(
            secondary_label_values=pd.Series([5, 1, 5, 1, 5, 1, 5, 1, 10]),
            ranks=pd.Series(list(range(1, 10))),
            click_rank=9,
            secondary_label="test_label")
        pd_testing.assert_series_equal(
            pd.Series(computed_metrics),
            pd.Series({
                "test_label_NDCG": 0.351,
                "test_label_failure_all": 1,
                "test_label_failure_any": 1,
                "test_label_failure_all_rank": 9,
                "test_label_failure_any_rank": 9,
                "test_label_failure_any_count": 8,
                "test_label_failure_any_fraction": 1.0
            }),
            check_less_precise=True)

    def test_compute_secondary_label_metrics_11(self):
        computed_metrics = compute_secondary_label_metrics(
            secondary_label_values=pd.Series([1, 1, 1, 1, 10, 10, 10, 10, 5]),
            ranks=pd.Series(list(range(1, 10))),
            click_rank=9,
            secondary_label="test_label")
        pd_testing.assert_series_equal(
            pd.Series(computed_metrics),
            pd.Series({
                "test_label_NDCG": 0.546,
                "test_label_failure_all": 0,
                "test_label_failure_any": 1,
                "test_label_failure_all_rank": 0,
                "test_label_failure_any_rank": 9,
                "test_label_failure_any_count": 4,
                "test_label_failure_any_fraction": 0.5
            }),
            check_less_precise=True)

    def test_compute_secondary_label_metrics_12(self):
        computed_metrics = compute_secondary_label_metrics(
            secondary_label_values=pd.Series([10, 10, 10, 10, 1, 1, 1, 1, 5]),
            ranks=pd.Series(list(range(1, 10))),
            click_rank=9,
            secondary_label="test_label")
        pd_testing.assert_series_equal(
            pd.Series(computed_metrics),
            pd.Series({
                "test_label_NDCG": 0.999,
                "test_label_failure_all": 0,
                "test_label_failure_any": 1,
                "test_label_failure_all_rank": 0,
                "test_label_failure_any_rank": 9,
                "test_label_failure_any_count": 4,
                "test_label_failure_any_fraction": 0.5
            }),
            check_less_precise=True)

    def test_compute_secondary_label_metrics_13(self):
        computed_metrics = compute_secondary_label_metrics(
            secondary_label_values=pd.Series([1, 10, 1, 10, 1, 10, 1, 10, 5]),
            ranks=pd.Series(list(range(1, 10))),
            click_rank=9,
            secondary_label="test_label")
        pd_testing.assert_series_equal(
            pd.Series(computed_metrics),
            pd.Series({
                "test_label_NDCG": 0.678,
                "test_label_failure_all": 0,
                "test_label_failure_any": 1,
                "test_label_failure_all_rank": 0,
                "test_label_failure_any_rank": 9,
                "test_label_failure_any_count": 4,
                "test_label_failure_any_fraction": 0.5
            }),
            check_less_precise=True)

    def test_compute_secondary_label_metrics_14(self):
        computed_metrics = compute_secondary_label_metrics(
            secondary_label_values=pd.Series([10, 1, 10, 1, 10, 1, 10, 1, 5]),
            ranks=pd.Series(list(range(1, 10))),
            click_rank=9,
            secondary_label="test_label")
        pd_testing.assert_series_equal(
            pd.Series(computed_metrics),
            pd.Series({
                "test_label_NDCG": 0.867,
                "test_label_failure_all": 0,
                "test_label_failure_any": 1,
                "test_label_failure_all_rank": 0,
                "test_label_failure_any_rank": 9,
                "test_label_failure_any_count": 4,
                "test_label_failure_any_fraction": 0.5
            }),
            check_less_precise=True)

    def test_compute_dcg(self):
        with self.subTest("Worst ordering of grade values"):
            self.assertTrue(np.isclose(compute_dcg([1., 2., 3.]), 4.261, atol=3))

        with self.subTest("Best ordering of grade values"):
            self.assertTrue(np.isclose(compute_dcg([3., 2., 1.]), 5.761, atol=3))

        with self.subTest("Equal grade values"):
            self.assertTrue(np.isclose(compute_dcg([1., 1., 1.]), 2.130, atol=3))

        with self.subTest("Zero grade values"):
            self.assertTrue(np.isclose(compute_dcg([0., 0., 0.]), 1.065, atol=3))

    def test_compute_ndcg(self):
        with self.subTest("Worst ordering of grade values"):
            self.assertTrue(np.isclose(compute_ndcg([1., 2., 3.]), 0.739, atol=3))

        with self.subTest("Best ordering of grade values"):
            self.assertTrue(np.isclose(compute_ndcg([3., 2., 1.]), 1., atol=3))

        with self.subTest("Equal grade values"):
            self.assertTrue(np.isclose(compute_ndcg([1., 1., 1.]), 1., atol=3))

        with self.subTest("Zero grade values"):
            self.assertTrue(np.isclose(compute_ndcg([0., 0., 0.]), 1., atol=3))
