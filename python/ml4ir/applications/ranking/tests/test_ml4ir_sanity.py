import unittest
import warnings
import pandas as pd
import numpy as np
import pathlib
from testfixtures import TempDirectory
import sys
sys.path.insert(0, '/Users/mohamed.m/Documents/work/projects/ml4ir_sanity_tests/python')
from ml4ir.applications.ranking.pipeline import RankingPipeline
from ml4ir.applications.ranking.config.parse_args import get_args

warnings.filterwarnings("ignore")


def calculate_mrr(y_pred, y_true):
    """
    Calculates the MRR given the predicted scores (y_pred) and the actual clicks (y_true).
    """
    if sum(y_true) < 1:
        return 0
    clickedDocIdxs = [i for i, x in enumerate(y_true) if x == 1.0]
    mrr = 0
    sortedPred = sorted(y_pred)[::-1]
    for clickedDocIdx in clickedDocIdxs:
        clickedDocScore = y_pred[clickedDocIdx]
        # ties are resolved with the worst case (i.e. the correct doc will have the lowest place among other docs with same score)
        equalScoreCount = sortedPred.count(clickedDocScore)
        predictedRank = sortedPred.index(clickedDocScore) + equalScoreCount - 1.0
        mrr += 1.0 / (predictedRank + 1.0)
    mrr = mrr / len(clickedDocIdxs)
    return mrr


def predict(model, documents):
    """
    Given a linear model and a matrix of documents, returns the prediction score for each document.
    """
    return np.dot(documents, model)


def check_mrr(df, model):
    """
    calculated MRR for a dataset given a model.
    """
    featureCount = len(model)
    MRR = 0
    queries = df.groupby('query_id')
    for qid, group in queries:
        X = [group['f' + str(i)] for i in range(featureCount)]
        pred = predict(model, np.array(X).T)
        mrr = calculate_mrr(pred, group['clicked'])
        MRR += mrr
    return MRR / len(queries)


def ml4ir_sanity_pipeline(df, perceptron, log_reg, working_dir, log_dir, n_features):
    """
    Train ml4ir on the passed data and calculate the MRR for ml4ir, perceptron and logistic regression models.
    """
    df.to_csv(working_dir / 'train' / 'data.csv')
    df.to_csv(working_dir / 'validation' / 'data.csv')
    df.to_csv(working_dir / 'test' / 'data.csv')

    fconfig_name = "feature_config_sanity_tests_" + str(n_features) + "_features.yaml"
    feature_config_file = pathlib.Path(__file__).parent / "data" / "configs" / fconfig_name
    model_config_file = pathlib.Path(__file__).parent / "data" / "configs" / "model_config_sanity_tests.yaml"
    train_ml4ir(working_dir.as_posix(), feature_config_file.as_posix(), model_config_file.as_posix(), log_dir.as_posix())

    ml4ir_weights = pd.read_csv("models/test_command_line/coefficients.csv")["weight"].tolist()
    ml4ir_weights = np.array(ml4ir_weights)
    ml4ir_mrr = check_mrr(df, ml4ir_weights)
    perceptron_mrr = check_mrr(df, perceptron)
    log_regression_mrr = check_mrr(df, log_reg)
    return ml4ir_mrr, perceptron_mrr, log_regression_mrr


def train_ml4ir(data_dir, feature_config, model_config, logs_dir):
    """
    Train a pointwise ranker, listwise loss model using ml4ir
    """
    argv = ['--data_dir', data_dir, '--feature_config', feature_config, '--loss_type', "listwise", '--scoring_type',
            "listwise",
            '--run_id', 'test_command_line', '--data_format', 'csv', '--execution_mode', 'train_inference_evaluate',
            '--loss_key', 'rank_one_listnet',
            '--num_epochs', '150', '--model_config', model_config, '--batch_size', '1', '--logs_dir', logs_dir,
            '--max_sequence_size', '25', '--train_pcent_split', '1.0', '--val_pcent_split', '-1', '--test_pcent_split',
            '-1', '--early_stopping_patience', '1000']
    args = get_args(argv)
    rp = RankingPipeline(args=args)
    rp.run()


def run_sanity_test(n_features, fname, perceptron, log_reg, working_dir, log_dir):
    """
    Runs sanity test for linear models.
    """
    df = pd.read_csv(pathlib.Path(__file__).parent / "data" / "L1_sanity_tests" / fname)
    ml4ir_mrr, perceptron_mrr, log_regression_mrr = ml4ir_sanity_pipeline(df, perceptron, log_reg,
                                                                          working_dir, log_dir,
                                                                          n_features)
    assert ml4ir_mrr >= perceptron_mrr
    assert ml4ir_mrr >= log_regression_mrr


class TestML4IRSanity(unittest.TestCase):
    def setUp(self):
        self.dir = pathlib.Path(__file__).parent
        self.working_dir = TempDirectory()
        self.log_dir = self.working_dir.makedir('logs')
        self.working_dir.makedir('train')
        self.working_dir.makedir('test')
        self.working_dir.makedir('validation')

    def tearDown(self):
        TempDirectory.cleanup_all()

    def test_linear_ml4ir_sanity_1(self):
        run_sanity_test(n_features=2, fname="dataset1.csv",
                        perceptron=np.array([1.87212065, -0.00305068]),
                        log_reg=np.array([28.62071696, 1.18915853]),
                        working_dir=pathlib.Path(self.working_dir.path), log_dir=pathlib.Path(self.log_dir))

    def test_linear_ml4ir_sanity_2(self):
        run_sanity_test(n_features=2, fname="dataset2.csv",
                        perceptron=np.array([4.50209484, -0.80280452]),
                        log_reg=np.array([22.73585585, -3.94821153]),
                        working_dir=pathlib.Path(self.working_dir.path), log_dir=pathlib.Path(self.log_dir))

    def test_linear_ml4ir_sanity_3(self):
        run_sanity_test(n_features=5, fname="dataset3.csv",
                        perceptron=np.array([-1.27651475, -4.07647092, 8.23950305, 0.29241316, 3.24763417]),
                        log_reg=np.array([-1.67270377, -5.76088727, 8.36278576, -0.90878154, 3.47653204]),
                        working_dir=pathlib.Path(self.working_dir.path), log_dir=pathlib.Path(self.log_dir))

    def test_linear_ml4ir_sanity_4(self):
        run_sanity_test(n_features=2, fname="dataset4.csv",
                        perceptron=np.array([5.10535665, 1.44131417]),
                        log_reg=np.array([20.0954756, 4.69360163]),
                        working_dir=pathlib.Path(self.working_dir.path), log_dir=pathlib.Path(self.log_dir))

    def test_linear_ml4ir_sanity_5(self):
        run_sanity_test(n_features=2, fname="dataset5.csv",
                        perceptron=np.array([0.57435291, -0.99437351]),
                        log_reg=np.array([1.15593505, -0.93317691]),
                        working_dir=pathlib.Path(self.working_dir.path), log_dir=pathlib.Path(self.log_dir))

    def test_linear_ml4ir_sanity_6(self):
        run_sanity_test(n_features=10, fname="dataset6.csv",
                        perceptron=np.array(
                            [4.59994733, -3.56373965, -6.15935686, 0.87523846, -0.64231058, 2.15971991, 5.79875003,
                             -7.70152594, -0.07521741, 2.8817456]),
                        log_reg=np.array(
                            [-0.38064406, -0.27970534, 0.02775136, 0.25641926, 0.15413321, 0.29194965, 0.72707686,
                             0.24791729, -0.39367192, 0.4882174]),
                        working_dir=pathlib.Path(self.working_dir.path), log_dir=pathlib.Path(self.log_dir))

    def test_linear_ml4ir_sanity_7(self):
        run_sanity_test(n_features=2, fname="dataset7.csv",
                        perceptron=np.array([0.40127356, -0.43773627]),
                        log_reg=np.array([4.15630544, -1.09111369]),
                        working_dir=pathlib.Path(self.working_dir.path), log_dir=pathlib.Path(self.log_dir))

    def test_linear_ml4ir_sanity_8(self):
        run_sanity_test(n_features=10, fname="dataset8.csv",
                        perceptron=np.array(
                            [2.91798129, 4.24880336, 7.42919018, 2.49609694, -0.84988373, 0.43435823, -0.18953416,
                             2.23129287, -0.67951411, -0.63925108]),
                        log_reg=np.array(
                            [-0.14472192, -0.22594271, 0.62703883, 0.16002515, 0.17084088, -0.22872226, 0.89200279,
                             0.06297475, 0.70470567, -0.19396659]),
                        working_dir=pathlib.Path(self.working_dir.path), log_dir=pathlib.Path(self.log_dir))


if __name__ == "__main__":
    unittest.main()
