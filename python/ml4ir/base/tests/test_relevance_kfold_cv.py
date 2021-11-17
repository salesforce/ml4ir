import unittest
import warnings
import pandas as pd
import numpy as np
import pathlib
import tensorflow as tf
from testfixtures import TempDirectory
from ml4ir.applications.ranking.pipeline import RankingPipeline
from ml4ir.applications.ranking.config.parse_args import get_args

warnings.filterwarnings("ignore")


def create_parse_args(data_dir, feature_config, model_config, logs_dir, num_folds, use_testset_in_folds):
    argv = ["--data_dir", data_dir,
            "--feature_config", feature_config,
            "--kfold", str(num_folds),
            "--include_testset_in_kfold", str(use_testset_in_folds),
            "--run_id", "testing_kfold_cs",
            "--data_format", "csv",
            "--execution_mode", "train_inference_evaluate",
            "--num_epochs", "1",
            "--model_config", model_config,
            "--batch_size", "4",
            "--logs_dir", logs_dir,
            "--max_sequence_size", "25",
            "--train_pcent_split", "1.0",
            "--val_pcent_split", "-1",
            "--test_pcent_split", "-1"]

    args = get_args(argv)
    return args


class TestML4IRKfoldCV(unittest.TestCase):
    """
    Test kfold cross validation
    """

    def setUp(self):
        self.dir = pathlib.Path(__file__).parent
        self.working_dir = TempDirectory()
        self.log_dir = self.working_dir.makedir('logs')
        self.working_dir.makedir('train')
        self.working_dir.makedir('test')
        self.working_dir.makedir('validation')
        tf.random.set_seed(123)
        np.random.seed(123)

    def tearDown(self):
        TempDirectory.cleanup_all()

    def setup_data(self, dataset_name, num_features, num_folds, use_testset_in_folds):
        """
        reading dataset and splitting it into 3 groups: train, validation and test.
        Parameters
        ----------
        dataset_name: str
            dataset to load (from the sanity tests datasets)
        num_features: int
            number of features in the dataset
        num_folds: int
            number of folds
        use_testset_in_folds: bool
            whether to include the testset in the merge
        """
        working_dir = pathlib.Path(self.working_dir.path)
        log_dir = pathlib.Path(self.log_dir)

        df = pd.read_csv(pathlib.Path(__file__).parent / "data" / "linear_sanity_tests" / dataset_name)
        fconfig_name = "feature_config_sanity_tests_" + str(num_features) + "_features.yaml"
        feature_config_file = pathlib.Path(__file__).parent / "data" / "configs" / fconfig_name
        model_config_file = pathlib.Path(__file__).parent / "data" / "configs" / "model_config_sanity_tests.yaml"

        queries = df.groupby('query_id')
        dfs = np.array_split(queries, 3)
        pd.concat([dfs[0][i][-1] for i in range(len(dfs[0]))]).to_csv(working_dir / 'train' / 'data.csv')
        pd.concat([dfs[1][i][-1] for i in range(len(dfs[1]))]).to_csv(working_dir / 'validation' / 'data.csv')
        pd.concat([dfs[2][i][-1] for i in range(len(dfs[2]))]).to_csv(working_dir / 'test' / 'data.csv')
        return create_parse_args(data_dir=working_dir.as_posix(),
                                 feature_config=feature_config_file.as_posix(),
                                 model_config=model_config_file.as_posix(),
                                 logs_dir=log_dir.as_posix(),
                                 num_folds=num_folds,
                                 use_testset_in_folds=use_testset_in_folds)

    def run_merge_datasets_test(self, dataset_name, num_features, num_folds, use_testset_in_folds, expected_num_queries):
        """
        With the specified input parameters, merge the datasets together and compare against the expected number of
        queries.

        Parameters
        ----------
        dataset_name: str
            dataset to load (from the sanity tests datasets)
        num_features: int
            number of features in the dataset
        num_folds: int
            number of folds
        use_testset_in_folds: bool
            whether to include the testset in the merge
        expected_num_queries: int
            expected number of queries ion the final merge.
        """
        args = self.setup_data(dataset_name, num_features, num_folds, use_testset_in_folds)

        rp = RankingPipeline(args=args)
        relevance_dataset = rp.get_kfold_relevance_dataset(args.kfold, args.include_testset_in_kfold,
                                                           read_data_sets=True)

        all_data = relevance_dataset.merge_datasets()
        query_ids = set([q[0]['query_id'].numpy()[0] for q in all_data])
        assert len(query_ids) == expected_num_queries

    def run_folds_creation_test(self, dataset_name, num_features, num_folds, use_testset_in_folds):
        """
        Read and merge datasets, then create folds. A successful fold creation should have different, non
        overlapping train, validation and test sets.

        Parameters
        ----------
        dataset_name: str
            dataset to load (from the sanity tests datasets)
        num_features: int
            number of features in the dataset
        num_folds: int
            number of folds
        use_testset_in_folds: bool
            whether to include the testset in the merge
        """
        args = self.setup_data(dataset_name, num_features, num_folds, use_testset_in_folds)

        rp = RankingPipeline(args=args)
        relevance_dataset = rp.get_kfold_relevance_dataset(args.kfold, args.include_testset_in_kfold,
                                                           read_data_sets=True)

        all_data = relevance_dataset.merge_datasets()
        for i in range(num_folds):
            fold_relevance_dataset = rp.get_kfold_relevance_dataset(args.kfold, args.include_testset_in_kfold,
                                                               read_data_sets=False)
            fold_relevance_dataset.create_folds(i, all_data, relevance_dataset)
            train = fold_relevance_dataset.train.unbatch()
            validation = fold_relevance_dataset.validation.unbatch()
            if use_testset_in_folds:
                test = fold_relevance_dataset.test.unbatch()
            train_qids = set([q[0]['query_id'].numpy()[0] for q in train])
            validation_qids = set([q[0]['query_id'].numpy()[0] for q in validation])
            if use_testset_in_folds:
                test_qids = set([q[0]['query_id'].numpy()[0] for q in test])
            if use_testset_in_folds:
                assert len(set.intersection(train_qids, test_qids)) == 0
                assert len(set.intersection(validation_qids, test_qids)) == 0
            assert len(set.intersection(train_qids, validation_qids)) == 0

    def test_merge_datasets_1(self):
        """
        Testing the dataset merge function with testset included in the merge.
        """
        dataset_name = "dataset1.csv"
        num_features = 2
        use_testset_in_folds = True
        expected_num_queries = 41
        num_folds = 5
        self.run_merge_datasets_test(dataset_name, num_features, num_folds, use_testset_in_folds, expected_num_queries)

    def test_merge_datasets_2(self):
        """
        Testing the dataset merge function without including testset in the merge
        """
        dataset_name = "dataset1.csv"
        num_features = 2
        use_testset_in_folds = False
        expected_num_queries = 28
        num_folds = 5
        self.run_merge_datasets_test(dataset_name, num_features, num_folds, use_testset_in_folds, expected_num_queries)

    def test_folds_creation_1(self):
        """
        Testing creating folds. Train, validation and test sets should be different with no over lap
        """
        dataset_name = "dataset1.csv"
        num_features = 2
        use_testset_in_folds = True
        num_folds = 5
        self.run_folds_creation_test(dataset_name, num_features, num_folds, use_testset_in_folds)

    def test_folds_creation_2(self):
        """
        Testing creating folds. Train, validation and test sets should be different with no over lap
        """
        dataset_name = "dataset1.csv"
        num_features = 2
        use_testset_in_folds = False
        num_folds = 5
        self.run_folds_creation_test(dataset_name, num_features, num_folds, use_testset_in_folds)

    def test_folds_creation_3(self):
        """
        Testing creating folds. Train, validation and test sets should be different with no over lap
        """
        dataset_name = "dataset1.csv"
        num_features = 2
        use_testset_in_folds = False
        num_folds = 10
        self.run_folds_creation_test(dataset_name, num_features, num_folds, use_testset_in_folds)

    def test_folds_creation_4(self):
        """
        Testing creating folds. Train, validation and test sets should be different with no over lap
        """
        dataset_name = "dataset1.csv"
        num_features = 2
        use_testset_in_folds = False
        num_folds = 10
        self.run_folds_creation_test(dataset_name, num_features, num_folds, use_testset_in_folds)

    def test_folds_creation_5(self):
        """
        Testing creating folds. Train, validation and test sets should be different with no over lap
        """
        dataset_name = "dataset2.csv"
        num_features = 2
        use_testset_in_folds = True
        num_folds = 10
        self.run_folds_creation_test(dataset_name, num_features, num_folds, use_testset_in_folds)

    def test_folds_creation_6(self):
        """
        Testing creating folds. Train, validation and test sets should be different with no over lap
        """
        dataset_name = "dataset2.csv"
        num_features = 2
        use_testset_in_folds = False
        num_folds = 10
        self.run_folds_creation_test(dataset_name, num_features, num_folds, use_testset_in_folds)


if __name__ == "__main__":
    unittest.main()
