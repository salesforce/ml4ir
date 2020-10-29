import unittest
import warnings
from ml4ir.base.features.preprocessing import *
from ml4ir.base.model.optimizers import cyclic_learning_rate

warnings.filterwarnings("ignore")

INPUT_DIR = "/ml4ir/applications/ranking/tests/data/"
KEEP_ADDITIONAL_INFO = 0
NON_ZERO_FEATURES_ONLY = 0


class TestCyclicLr(unittest.TestCase):
    def test_cyclic_tri_lr(self):
        """Test a triangular cyclic learning rate"""
        gold = [0.1, 0.28000003, 0.45999995, 0.64000005, 0.81999993, 1.0, 0.81999993, 0.64000005, 0.45999995, 0.28000003,
                0.1, 0.28000003, 0.46000007, 0.6399999, 0.81999993, 1.0, 0.81999993, 0.6399999, 0.46000007, 0.28000003,
                0.1, 0.27999982, 0.46000007, 0.6399999, 0.8200002]
        lrs = cyclic_learning_rate.TriangularCyclicalLearningRate(
            initial_learning_rate=0.1,
            maximal_learning_rate=1.0,
            step_size=5,
        )
        for i in range(25):
            assert round(float(lrs(i).numpy()), 5) == round(gold[i], 5)

    def test_cyclic_tri2_lr(self):
        """Test a triangular2 cyclic learning rate"""
        gold = [0.1, 0.28000003, 0.45999995, 0.64000005, 0.81999993, 1.0, 0.81999993, 0.64000005, 0.45999995, 0.28000003, 0.1,
                0.19000003, 0.28000003, 0.36999995, 0.45999995, 0.55, 0.45999995, 0.36999995, 0.28000003, 0.19000003, 0.1,
                0.14499995, 0.19000003, 0.23499998, 0.28000003]
        lrs = cyclic_learning_rate.Triangular2CyclicalLearningRate(
            initial_learning_rate=0.1,
            maximal_learning_rate=1.0,
            step_size=5,
        )
        for i in range(25):
            assert round(float(lrs(i).numpy()), 5) == round(gold[i], 5)

    def test_cyclic_exp_lr(self):
        """Test a exponential cyclic learning rate"""
        gold = [0.1, 0.26200002, 0.39159995, 0.49366, 0.5723919, 0.631441, 0.48263744, 0.35828033, 0.25496817, 0.1697357, 0.1,
                0.15648592, 0.20167467, 0.23726073, 0.2647129, 0.285302, 0.23341745, 0.19005677, 0.15403408, 0.12431534, 0.1,
                0.1196954, 0.13545176, 0.14785986, 0.15743186]
        lrs = cyclic_learning_rate.ExponentialCyclicalLearningRate(
            initial_learning_rate=0.1,
            maximal_learning_rate=1.0,
            step_size=5,
            gamma=0.9,
        )
        for i in range(25):
            assert round(float(lrs(i).numpy()), 5) == round(gold[i], 5)

    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()
