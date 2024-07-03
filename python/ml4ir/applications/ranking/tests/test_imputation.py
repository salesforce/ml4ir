import unittest

import numpy as np
from scipy.stats import zscore

from ml4ir.applications.ranking.model.layers.imputation import QueryMinImputation


class TestQueryMinImputation(unittest.TestCase):

    def test_query_min_imputation(self):
        query_min_imputation = QueryMinImputation()

        x = np.array([
            [
                [1., 3., 0.], 
                [2., 4., 5.], 
                [0., 0., 2.]
            ],
            [
                [10., 25., 0.],
                [20., 30., 0.],
                [0., 0., 0.]
            ]])
        actual_min_imp = query_min_imputation(x)

        expected_min_imp = np.array([
            [
                [1., 3., 2.],
                [2., 4., 5.],
                [1., 3., 2.]
            ],
            [
                [10., 25., 0.],
                [20., 30., 0.],
                [10., 25., 0.]
            ]])
        self.assertTrue(np.isclose(expected_min_imp, actual_min_imp).all())

    def test_query_min_imputation_missing_value(self):
        query_min_imputation = QueryMinImputation(missing_value=-1.)

        x = np.array([
            [
                [1., 3., -1], 
                [2., 4., 5.], 
                [-1, -1, 2.]
            ],
            [
                [10., 25., -1],
                [20., 30., 0.],
                [-1, 0., 0.]
            ]])
        actual_min_imp = query_min_imputation(x)

        expected_min_imp = np.array([
            [
                [1., 3., 2.],
                [2., 4., 5.],
                [1., 3., 2.]
            ],
            [
                [10., 25., 0.],
                [20., 30., 0.],
                [10., 0., 0.]
            ]])
        self.assertTrue(np.isclose(expected_min_imp, actual_min_imp).all())