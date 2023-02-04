import unittest
from unittest.mock import patch

import pandas as pd
import numpy as np
from pandas import testing as pd_testing

from ml4ir.applications.ranking.model.metrics.helpers.aux_metrics_helper import *


class ComputeAuxMetricsTest(unittest.TestCase):
    """Test suite for ml4ir.applications.ranking.model.metrics.metrics_helper"""

    def test_compute_aux_metrics(self):
        """Test the auxiliary metrics computation in aux_metrics_helper with different input values"""
        with self.subTest("Test Case: 1"):
            computed_metrics = compute_aux_metrics(
                aux_label_values=pd.Series([10, 10, 10, 10, 10, 10, 10, 10, 1]),
                ranks=pd.Series(list(range(1, 10))),
                click_rank=9)
            pd_testing.assert_series_equal(
                pd.Series(computed_metrics),
                pd.Series({
                    "AuxAllFailure": 0,
                    "AuxIntrinsicFailure": 0.
                }),
                check_less_precise=True)

        with self.subTest("Test Case: 2"):
            computed_metrics = compute_aux_metrics(
                aux_label_values=pd.Series([1, 1, 1, 1, 1, 1, 1, 1, 10]),
                ranks=pd.Series(list(range(1, 10))),
                click_rank=9)

            pd_testing.assert_series_equal(
                pd.Series(computed_metrics),
                pd.Series({
                    "AuxAllFailure": 1,
                    "AuxIntrinsicFailure": 0.69608
                }),
                check_less_precise=True)

        with self.subTest("Test Case: 3"):
            computed_metrics = compute_aux_metrics(
                aux_label_values=pd.Series([10, 10, 10, 1]),
                ranks=pd.Series(list(range(1, 5))),
                click_rank=4)
            pd_testing.assert_series_equal(
                pd.Series(computed_metrics),
                pd.Series({
                    "AuxAllFailure": 0,
                    "AuxIntrinsicFailure": 0.
                }),
                check_less_precise=True)

        with self.subTest("Test Case: 4"):
            computed_metrics = compute_aux_metrics(
                aux_label_values=pd.Series([1, 1, 1, 10]),
                ranks=pd.Series(list(range(1, 5))),
                click_rank=4)
            pd_testing.assert_series_equal(
                pd.Series(computed_metrics),
                pd.Series({
                    "AuxAllFailure": 1,
                    "AuxIntrinsicFailure": 0.568
                }),
                check_less_precise=True)

        with self.subTest("Test Case: 5"):
            computed_metrics = compute_aux_metrics(
                aux_label_values=pd.Series([1, 1, 1, 5]),
                ranks=pd.Series(list(range(1, 5))),
                click_rank=4)
            pd_testing.assert_series_equal(
                pd.Series(computed_metrics),
                pd.Series({
                    "AuxAllFailure": 1,
                    "AuxIntrinsicFailure": 0.525
                }),
                check_less_precise=True)

        with self.subTest("Test Case: 6"):
            computed_metrics = compute_aux_metrics(
                aux_label_values=pd.Series([5, 5, 5, 10]),
                ranks=pd.Series(list(range(1, 5))),
                click_rank=4)
            pd_testing.assert_series_equal(
                pd.Series(computed_metrics),
                pd.Series({
                    "AuxAllFailure": 1.0,
                    "AuxIntrinsicFailure": 0.52723
                }),
                check_less_precise=True)

        with self.subTest("Test Case: 7"):
            computed_metrics = compute_aux_metrics(
                aux_label_values=pd.Series([1, 1, 1, 1, 5, 5, 5, 5, 10]),
                ranks=pd.Series(list(range(1, 10))),
                click_rank=9)
            pd_testing.assert_series_equal(
                pd.Series(computed_metrics),
                pd.Series({
                    "AuxAllFailure": 1,
                    "AuxIntrinsicFailure": 0.674
                }),
                check_less_precise=True)

        with self.subTest("Test Case: 8"):
            computed_metrics = compute_aux_metrics(
                aux_label_values=pd.Series([5, 5, 5, 5, 1, 1, 1, 1, 10]),
                ranks=pd.Series(list(range(1, 10))),
                click_rank=9)
            pd_testing.assert_series_equal(
                pd.Series(computed_metrics),
                pd.Series({
                    "AuxAllFailure": 1,
                    "AuxIntrinsicFailure": 0.6416
                }),
                check_less_precise=True)

        with self.subTest("Test Case: 9"):
            computed_metrics = compute_aux_metrics(
                aux_label_values=pd.Series([1, 5, 1, 5, 1, 5, 1, 5, 10]),
                ranks=pd.Series(list(range(1, 10))),
                click_rank=9)
            pd_testing.assert_series_equal(
                pd.Series(computed_metrics),
                pd.Series({
                    "AuxAllFailure": 1,
                    "AuxIntrinsicFailure": 0.66452
                }),
                check_less_precise=True)

        with self.subTest("Test Case: 10"):
            computed_metrics = compute_aux_metrics(
                aux_label_values=pd.Series([5, 1, 5, 1, 5, 1, 5, 1, 10]),
                ranks=pd.Series(list(range(1, 10))),
                click_rank=9)
            pd_testing.assert_series_equal(
                pd.Series(computed_metrics),
                pd.Series({
                    "AuxAllFailure": 1,
                    "AuxIntrinsicFailure": 0.651047
                }),
                check_less_precise=True)

        with self.subTest("Test Case: 11"):
            computed_metrics = compute_aux_metrics(
                aux_label_values=pd.Series([1, 1, 1, 1, 10, 10, 10, 10, 5]),
                ranks=pd.Series(list(range(1, 10))),
                click_rank=9)
            pd_testing.assert_series_equal(
                pd.Series(computed_metrics),
                pd.Series({
                    "AuxAllFailure": 0,
                    "AuxIntrinsicFailure": 0.455
                }),
                check_less_precise=True)

        with self.subTest("Test Case: 12"):
            computed_metrics = compute_aux_metrics(
                aux_label_values=pd.Series([10, 10, 10, 10, 1, 1, 1, 1, 5]),
                ranks=pd.Series(list(range(1, 10))),
                click_rank=9)
            pd_testing.assert_series_equal(
                pd.Series(computed_metrics),
                pd.Series({
                    "AuxAllFailure": 0,
                    "AuxIntrinsicFailure": 0.001
                }),
                check_less_precise=True)

        with self.subTest("Test Case: 13"):
            computed_metrics = compute_aux_metrics(
                aux_label_values=pd.Series([1, 10, 1, 10, 1, 10, 1, 10, 5]),
                ranks=pd.Series(list(range(1, 10))),
                click_rank=9)
            pd_testing.assert_series_equal(
                pd.Series(computed_metrics),
                pd.Series({
                    "AuxAllFailure": 0,
                    "AuxIntrinsicFailure": 0.3224
                }),
                check_less_precise=True)

        with self.subTest("Test Case: 14"):
            computed_metrics = compute_aux_metrics(
                aux_label_values=pd.Series([10, 1, 10, 1, 10, 1, 10, 1, 5]),
                ranks=pd.Series(list(range(1, 10))),
                click_rank=9)
            pd_testing.assert_series_equal(
                pd.Series(computed_metrics),
                pd.Series({
                    "AuxAllFailure": 0,
                    "AuxIntrinsicFailure": 0.1335
                }),
                check_less_precise=True)

    def test_compute_aux_label_metrics_invalid_click(self):
        """Testing compute_aux_metrics method with invalid click values"""
        computed_metrics = compute_aux_metrics(
            aux_label_values=pd.Series([10, 1, 10, 1, 10, 1, 10, 1, 5]),
            ranks=pd.Series(list(range(1, 10))),
            click_rank=15)
        pd_testing.assert_series_equal(
            pd.Series(computed_metrics),
            pd.Series({
                "AuxAllFailure": 0,
                "AuxIntrinsicFailure": 0.133459
            }),
            check_less_precise=True)

    @patch("ml4ir.applications.ranking.model.metrics.helpers.aux_metrics_helper.compute_aux_metrics")
    def test_compute_aux_metrics_on_query_group(self, mock_compute_aux_metrics):
        """Testing compute_aux_metrics_on_query_group wrapper for compute_aux_metrics"""
        query_group = pd.DataFrame({
            "old_rank": [1, 2, 3, 4, 5],
            "new_rank": [3, 2, 1, 5, 4],
            "click": [0, 0, 1, 0, 0],
            "aux_label": [5, 5, 5, 2, 2]
        })
        mock_compute_aux_metrics.return_value = {}
        compute_aux_metrics_on_query_group(
            query_group=query_group,
            label_col="click",
            old_rank_col="old_rank",
            new_rank_col="new_rank",
            aux_label="aux_label")

        assert mock_compute_aux_metrics.call_count == 2 * 1

        call_args = [args[1] for args in mock_compute_aux_metrics.call_args_list]
        i = 0
        for state in ["old", "new"]:
            assert pd.Series.equals(
                call_args[i]["aux_label_values"], query_group["aux_label"])
            assert pd.Series.equals(call_args[i]["ranks"], query_group["{}_rank".format(state)])
            assert call_args[i]["click_rank"] == query_group[query_group["click"]
                                                             == 1]["{}_rank".format(state)].values[0]
            assert call_args[i]["prefix"] == "{}_".format(state)

            i += 1

    def test_compute_aux_metrics_on_query_group_invalid_click(self):
        """Testing compute_aux_metrics_on_query_group wrapper function with invalid click values"""
        query_group = pd.DataFrame({
            "old_rank": [1, 2, 3, 4, 5],
            "new_rank": [3, 2, 1, 5, 4],
            "click": [0, 0, 0, 0, 0],
            "aux_label": [5, 5, 5, 2, 2]
        })

        aux_metrics = compute_aux_metrics_on_query_group(
            query_group=query_group,
            label_col="click",
            old_rank_col="old_rank",
            new_rank_col="new_rank",
            aux_label="aux_label")
        intrinsic_failure_rows = aux_metrics.index.str.contains("IntrinsicFailure")
        self.assertEqual((aux_metrics[intrinsic_failure_rows] < 1).sum(),
                         len(aux_metrics[intrinsic_failure_rows]),
                         "IntrinsicFailure should be <1 in all cases")
        self.assertEqual(aux_metrics[~intrinsic_failure_rows].sum(),
                         0,
                         "All metrics should have default values")

    def test_compute_dcg(self):
        """Test DCG computation"""
        with self.subTest("Worst ordering of grade values"):
            self.assertTrue(np.isclose(compute_dcg([1., 2., 3.]), 6.39278, atol=3))

        with self.subTest("Best ordering of grade values"):
            self.assertTrue(np.isclose(compute_dcg([3., 2., 1.]), 9.392789, atol=3))

        with self.subTest("Equal grade values"):
            self.assertTrue(np.isclose(compute_dcg([1., 1., 1.]), 2.1309, atol=3))

        with self.subTest("Zero grade values"):
            self.assertTrue(np.isclose(compute_dcg([0., 0., 0.]), 1.065, atol=3))

    def test_compute_ndcg(self):
        """Test NDCG computation"""
        with self.subTest("Worst ordering of grade values"):
            self.assertTrue(np.isclose(compute_ndcg([1., 2., 3.]), 0.6806, atol=3))

        with self.subTest("Best ordering of grade values"):
            self.assertTrue(np.isclose(compute_ndcg([3., 2., 1.]), 1., atol=3))

        with self.subTest("Equal grade values"):
            self.assertTrue(np.isclose(compute_ndcg([1., 1., 1.]), 1., atol=3))
