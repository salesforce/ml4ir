from tensorflow.keras import layers

from ml4ir.base.io.file_io import FileIO


class BaseFeatureLayerOp(layers.Layer):
    """Abstract feature layer operation class"""

    def __init__(self, feature_info: dict, file_io: FileIO, **kwargs):
        """
        Initialize the feature layer

        Parameters
        ----------
        feature_info : dict
            Dictionary representing the feature_config for the input feature
        file_io : FileIO object
            FileIO handler object for reading and writing
        """
        super().__init__(**kwargs)

        self.feature_layer_args = feature_info["feature_layer_info"]["args"]
        self.feature_layer_args["node_name"] = feature_info.get("node_name", feature_info["name"])
        self.feature_name = self.feature_layer_args["node_name"]

        self.file_io = file_io

    def get_config(self):
        """
        Get the layer configuration.
        Used for serialization with Functional Keras model
        """
        config = super().get_config()
        config.update(self.feature_layer_args)
        return config