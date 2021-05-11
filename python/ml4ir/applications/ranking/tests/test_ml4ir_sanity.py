import unittest
import warnings
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.utils import check_random_state
from ml4ir.applications.ranking.pipeline import RankingPipeline
from ml4ir.applications.ranking.config.parse_args import get_args
import shutil
import pathlib
import random
from argparse import Namespace
warnings.filterwarnings("ignore")



def create_ml4ir_dataframe(X, Y):
    """
    Takes the generated data (X: data features, Y: clicked) and formulate a dataframe consumable by ml4ir
    """
    def add_features_to_record(X, xid, r, label):
        for i in range(len(X[xid])):
            r['f' + str(i)] = X[xid][i]
        r['clicked'] = label

    clicked_docs = X[Y == 1]
    nonclicked_docs = X[Y == 0]
    consumed_clicked_docs = 0
    consumed_nonclicked_docs = 0
    rows = []
    query_count = 0
    document_count = 0
    while consumed_clicked_docs < len(clicked_docs) and consumed_nonclicked_docs < len(nonclicked_docs):
        qid = 'query'+str(query_count)
        query_count += 1
        r = {'query_id':qid}
        docs_count = random.randint(2, 25)
        r['record_id'] = 'record'+str(document_count)
        r['fr'] =docs_count
        add_features_to_record(clicked_docs, consumed_clicked_docs, r, 1)
        consumed_clicked_docs += 1
        document_count += 1
        rows.append(r)
        for i in range(docs_count-1):
            if consumed_nonclicked_docs < len(nonclicked_docs):
                r = {'query_id': qid}
                r['fr'] = i+1
                r['record_id'] = 'record' + str(document_count)
                add_features_to_record(nonclicked_docs, consumed_nonclicked_docs, r, 0)
                consumed_nonclicked_docs += 1
                document_count += 1
                rows.append(r)
            else:
                break
    return pd.DataFrame(rows)


def generate_data(args):
    """
    Generate data with the specified parameters
    - n_classes: Number of classes in the datasets. Should be 2 (clicked, not clicked)
    - n_samples: Number of samples to generate
    - n_features: Number of features per sample
    - n_redundant: The number of redundant features
    - n_informative: The number of informative (Non-redundant) features
    - n_clusters_per_class: The number of clusters per class.
    - flip_y: The fraction of samples whose class is assigned randomly. Larger
        values introduce noise in the labels and make the classification
        task harder.
    - class_sep: Larger values spread out the clusters/classes and make the classification task easier.
    - seed: Random seed
    Returns: the best seprabale data set by a Perceptorn.
    """
    separable = False
    seed = args.seed
    rng = check_random_state(args.seed)
    best_accuracy = 0
    best_accuracy_data = None
    trials = 1000
    while not separable and trials > 0:
        samples = datasets.make_classification(n_classes=args.n_classes, n_samples=args.n_samples, n_features=args.n_features,
                                               n_redundant=args.n_redundant, n_informative=args.n_informative,
                                               n_clusters_per_class=args.n_clusters_per_class, flip_y=args.flip_y,
                                               class_sep=args.class_sep, random_state=rng)
        p = Perceptron(fit_intercept=True, max_iter=1000, verbose=0, random_state=seed, validation_fraction=0.0001)
        p.fit(samples[0], samples[1])
        acc = p.score(samples[0], samples[1])
        if acc > best_accuracy:
            best_accuracy = acc
            best_accuracy_data = samples
        acc = p.score(samples[0], samples[1])
        if acc == 1.0:
            break
        trials -= 1
    return best_accuracy_data[0], best_accuracy_data[1]


def calculate_MRR(prediction, y):
    """
    Calculates the MRR given the predicted scores (prediction) and the actual clicks (y).
    """
    if sum(y) < 1:
        return 0
    pred = prediction
    clickedDocIdxs = [i for i, x in enumerate(y) if x == 1.0]
    MRR = 0
    sortedPred = sorted(pred)[::-1]
    for clickedDocIdx in clickedDocIdxs:
        clickedDocScore = pred[clickedDocIdx]
        # ties are resolved with the worst case (i.e. the correct doc will have the lowest place among other docs with same score)
        equalScoreCount = sortedPred.count(clickedDocScore)
        predictedRank = sortedPred.index(clickedDocScore) + equalScoreCount - 1.0
        MRR += 1.0 / (predictedRank + 1.0)
    MRR = MRR / len(clickedDocIdxs)
    return MRR


def check_MRR(df, model, featureCount=2):
    """
    calculated MRR for a dataset given a model.
    """
    MRR = 0
    queries = df.groupby('query_id')
    for qid, group in queries:
        X = [group['f' + str(i)] for i in range(featureCount)]
        pred = model.decision_function(np.array(X).T)
        mrr = calculate_MRR(pred, group['clicked'])
        MRR += mrr
    return MRR/len(queries)


def check_accuracy(df, model, featureCount=2):
    X = [df['f'+str(i)] for i in range(featureCount)]
    Y = df['clicked']
    acc = model.score(np.array(X).T, Y)
    return acc


def train_sklearn_model(df, model, featureCount=2):
    """
    Trains a sklearn model (Specifically, logistic regression and Perceptron)
    """
    X = [df['f' + str(i)] for i in range(featureCount)]
    Y = df['clicked']
    model.fit(np.array(X).T, Y)
    for i in range(featureCount):
        model.coef_[0][i] = 1.0
    if model.fit_intercept:
        model.intercept_ = 1.0
    else:
        model.intercept_ = 0
    model.fit(np.array(X).T, Y)
    return model


def generate(args):
    """
    create an ml4ir consumable dataframe after generating dataset.
    """
    X, Y = generate_data(args)
    df = create_ml4ir_dataframe(X, Y)
    return df


def wrap_linear_model_in_sklearn_model(df, W):
    """
    Wrap plain weights and bias in a dummy sklearn model
    """
    wrapper_model = Perceptron(fit_intercept=True, max_iter=1, verbose=0, validation_fraction=0.0001)
    X = [df['f' + str(i)] for i in range(len(W)-1)]
    Y = df['clicked']
    wrapper_model.fit(np.array(X).T, Y)
    for i in range(len(W)-1):
        wrapper_model.coef_[0][i] = W[i]
    wrapper_model.intercept_ = W[-1]
    return wrapper_model


class TestML4IRSanity(unittest.TestCase):
    def setUp(self):
        self.dir = pathlib.Path(__file__).parent
        self.p = pathlib.Path(self.dir) / "data" /"ml4ir_sanity_test_working_dir"
        self.p.mkdir(parents=True, exist_ok=True)
        self.log_dir = self.p / 'logs'
        self.log_dir.mkdir(parents=True, exist_ok=True)
        train = self.p / 'train'
        train.mkdir(parents=True, exist_ok=True)
        test = self.p / 'test'
        test.mkdir(parents=True, exist_ok=True)
        validation = self.p / 'validation'
        validation.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.p)

    def test_ml4ir_sanity_1(self):
        args = Namespace(n_classes=2, n_samples=1000, n_features=2, n_redundant=0, n_informative=2,
                         n_clusters_per_class=2, flip_y=-1, class_sep=1.0, seed=13)
        self.test_ml4ir_sanity_pipeline(args)

    def test_ml4ir_sanity_2(self):
        args = Namespace(n_classes=2, n_samples=500, n_features=5, n_redundant=0, n_informative=5,
                         n_clusters_per_class=3, flip_y=-1, class_sep=2.0, seed=13)
        self.test_ml4ir_sanity_pipeline(args)

    def test_ml4ir_sanity_3(self):
        args = Namespace(class_sep=0.1, flip_y=-1.0, n_classes=2, n_clusters_per_class=2, n_features=2, n_informative=2,
                         n_redundant=0, n_samples=1000, seed=1111)
        self.test_ml4ir_sanity_pipeline(args)

    def test_ml4ir_sanity_4(self):
        args = Namespace(class_sep=0.1, flip_y=-1.0, n_classes=2, n_clusters_per_class=5, n_features=10, n_informative=7, n_redundant=3,
                         n_samples=1000, seed=1111)
        self.test_ml4ir_sanity_pipeline(args)

    def test_ml4ir_sanity_pipeline(self, args):
        df = generate(args)
        df.to_csv(self.p / 'train' / 'data.csv')
        df.to_csv(self.p / 'validation' / 'data.csv')
        df.to_csv(self.p / 'test' / 'data.csv')

        fconfig_name = "feature_config_sanity_tests_" + str(args.n_features) + "_features.yaml"
        feature_config_file = pathlib.Path(__file__).parent / "data" / "configs" / fconfig_name
        model_config_file = pathlib.Path(__file__).parent / "data" / "configs" / "model_config_sanity_tests.yaml"
        self.train_ml4ir(self.p.as_posix(), feature_config_file.as_posix(), model_config_file.as_posix(), self.log_dir.as_posix())
        logisticReg = LogisticRegression(fit_intercept=False, random_state=args.seed, max_iter=100,
                                warm_start=True)
        perceptron = Perceptron(fit_intercept=False, max_iter=100, random_state=args.seed, validation_fraction=0.0001,
                       warm_start=True)
        ml4ir_weights = pd.read_csv("models/test_command_line/coefficients.csv")["weight"].tolist()
        ml4ir_weights.append(0.0)
        ml4ir = wrap_linear_model_in_sklearn_model(df, ml4ir_weights)
        train_sklearn_model(df, logisticReg, featureCount=args.n_features)
        train_sklearn_model(df, perceptron, featureCount=args.n_features)

        ml4ir_mrr = check_MRR(df, ml4ir, args.n_features)
        perceptron_mrr = check_MRR(df, perceptron, args.n_features)
        log_regression_mrr = check_MRR(df, logisticReg, args.n_features)
        assert ml4ir_mrr >= perceptron_mrr
        assert ml4ir_mrr >= log_regression_mrr

    def train_ml4ir(self, data_dir, feature_config, model_config, logs_dir):

        argv = ['--data_dir', data_dir, '--feature_config', feature_config, '--loss_type', "pointwise", '--scoring_type', "pointwise",
                '--run_id', 'test_command_line', '--data_format', 'csv', '--execution_mode', 'train_inference_evaluate', '--loss_key', 'sigmoid_cross_entropy',
                '--num_epochs', '150', '--model_config', model_config, '--batch_size', '16', '--logs_dir', logs_dir,
                '--max_sequence_size', '25', '--train_pcent_split', '1.0', '--val_pcent_split', '-1', '--test_pcent_split', '-1', '--early_stopping_patience', '1000']
        args = get_args(argv)
        rp = RankingPipeline(args=args)
        rp.run()





if __name__ == "__main__":
    unittest.main()
