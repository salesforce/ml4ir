import unittest
import ml4ir.base.config.dynamic_args as dynamic_args
from ml4ir.base.config.parse_args import RelevanceArgParser
from ml4ir.base.io.local_io import LocalIO

FEATURE_CONFIG_PATH = "ml4ir/applications/ranking/tests/data/configs/feature_config.yaml"
MODEL_CONFIG_PATH = "ml4ir/base/config/default_model_config.yaml"


class DynamicArgsTest(unittest.TestCase):

    def setUp(self):
        file_io = LocalIO()
        self.feature_config_dict = file_io.read_yaml(FEATURE_CONFIG_PATH)
        self.model_config_dict = file_io.read_yaml(MODEL_CONFIG_PATH)

    def test_dynamic_args(self):
        """
        Unit test the accepted dynamic args by argparse
        """
        parser = RelevanceArgParser()

        # Test empty dynamic args
        args = parser.parse_args([])
        assert "feature_config_custom" not in args
        assert "model_config_custom" not in args

        # Test feature config and model config custom args
        args = parser.parse_args([
            "--feature_config.features.query_text.feature_layer_info.args.encoding_size",
            "512",
            "--model_config.layers.first_dense.units",
            "1024"
        ])
        assert len(args.feature_config_custom) == 1
        assert len(args.model_config_custom) == 1
        assert args.feature_config_custom[
            "features.query_text.feature_layer_info.args.encoding_size"] == "512"
        assert args.model_config_custom[
            "layers.first_dense.units"] == "1024"

    def test_cast_dynamic_val(self):
        """
        Unit test the value casting method
        """
        # Test string casting
        assert isinstance(dynamic_args.cast_dynamic_val("abc"), str)
        assert isinstance(dynamic_args.cast_dynamic_val("123"), int)
        assert isinstance(dynamic_args.cast_dynamic_val("12.3"), float)
        assert isinstance(dynamic_args.cast_dynamic_val("[1, 2, 3]"), list)
        assert isinstance(dynamic_args.cast_dynamic_val(
            "{'a': 1, 'b': 2}"), dict)

        # Test no casting when input val is not string
        assert isinstance(dynamic_args.cast_dynamic_val(123), int)
        assert isinstance(dynamic_args.cast_dynamic_val(12.3), float)
        assert isinstance(dynamic_args.cast_dynamic_val([1, 2, 3]), list)
        assert isinstance(
            dynamic_args.cast_dynamic_val({'a': 1, 'b': 2}), dict)

    def test_override_with_dynamic_args(self):
        """
        Unit test the override_with_dynamic_args method

        Check if the dictionary of dynamic arguments are applied to the
        base dictionary
        """

        # Test feature config updates
        base_dict = self.feature_config_dict
        assert not base_dict["features"][4]["feature_layer_info"]["args"]["encoding_size"] == 1024
        assert not base_dict["features"][6]["feature_layer_info"]["args"]["embedding_size"] == 1024

        base_dict = dynamic_args.override_with_dynamic_args(
            base_dict,
            {
                "features.4.feature_layer_info.args.encoding_size": "1024",
                "features.domain_name.feature_layer_info.args.embedding_size": 1024
            })

        assert base_dict["features"][4]["feature_layer_info"]["args"]["encoding_size"] == 1024
        assert base_dict["features"][6]["feature_layer_info"]["args"]["embedding_size"] == 1024

        # Test model config updates
        base_dict = self.model_config_dict
        assert not base_dict["optimizer"]["key"] == "sgd"
        assert not base_dict["lr_schedule"]["learning_rate"] == 0.00005

        base_dict = dynamic_args.override_with_dynamic_args(
            base_dict,
            {
                "optimizer.key": "sgd",
                "lr_schedule.learning_rate": "0.00005"
            })

        assert base_dict["optimizer"]["key"] == "sgd"
        assert base_dict["lr_schedule"]["learning_rate"] == 0.00005

    def test_override_dict(self):
        """
        Unit test the override_dict method

        Check if a dictionary can be overriden with a value
        """
        # Test high level key
        assert dynamic_args.override_dict(
            {
                "key_0": "val_0",
                "key_1": "val_1"
            },
            "key_0",
            "val_x")["key_0"] == "val_x"

        # Test nested dict
        assert dynamic_args.override_dict(
            {
                "key_0": {
                    "key_0_0": "val_0_0",
                    "key_0_1": "val_0_1"
                },
                "key_1": "val_1"
            },
            "key_0.key_0_1",
            "val_x")["key_0"]["key_0_1"] == "val_x"

    def test_override_list(self):
        """
        Unit test the override_dict method

        Check if a list can be overriden with a value
        """
        assert dynamic_args.override_list(
            [
                "val_0",
                "val_1",
                "val_2"
            ],
            "2",
            "val_x"
        )[2] == "val_x"

        # Test updating by matching on `name`
        assert dynamic_args.override_list(
            [
                {"name": "val_0"},
                {"name": "val_1"}
            ],
            "val_0.name",
            "val_x"
        )[0]["name"] == "val_x"

        # Test updating by array index
        assert dynamic_args.override_list(
            [
                {"name": "val_0"},
                {"name": "val_1"}
            ],
            "0.name",
            "val_x"
        )[0]["name"] == "val_x"

    def test_new_arg(self):
        """
        Unit test adding a new key value argument
        """
        updated_dict = dynamic_args.override_dict(
            {
                "key_0": "val_0",
                "key_1": "val_1"
            },
            "key_2",
            "val_2")

        assert len(updated_dict) == 3
        assert updated_dict["key_2"] == "val_2"
