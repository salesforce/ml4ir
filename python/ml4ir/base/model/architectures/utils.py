from typing import Dict, Type
from functools import lru_cache
import json

import tensorflow as tf


@lru_cache(maxsize=None)
def get_keras_layer_subclasses() -> Dict[str, Type[tf.keras.layers.Layer]]:
    """Get mapping of {subclass-name: subclass} for all derivative classes of keras.layers.Layer"""

    def full_class_name(cls):
        """
        Get {package_name}.{class_name}

        Examples:
            - keras: keras.layers.merge.Concatenate, keras.layers.core.dense.Dense
            - ml4ir: ml4ir.base.features.feature_fns.utils.VocabLookup,
                     ml4ir.applications.ranking.model.losses.listwise_losses.RankOneListNet
        """
        module = cls.__module__
        if module == "builtins":
            return cls.__qualname__  # avoid outputs like 'builtins.str'
        return f"{module}.{cls.__qualname__}"

    def get_sub_classes(cls):
        """DFS traversal to get all subclasses of `cls` class"""
        for subcls in cls.__subclasses__():
            if subcls not in subclasses:
                # Handles duplicates
                subclasses[full_class_name(subcls)] = subcls
                get_sub_classes(subcls)

    subclasses = {}
    get_sub_classes(tf.keras.layers.Layer)

    return subclasses


def instantiate_keras_layer(current_layer_type: str, current_layer_args: Dict) -> tf.keras.layers.Layer:
    """
    Create and return an instance of `current_layer_type` with `current_layer_args` params
    If `current_layer_type` is not found in the set of subclasses of keras.layers.Layer, throws KeyError

    Parameters
    ----------
    current_layer_type: str
        Refers to class name inheriting directly or indirectly from keras.layers.Layer to be instantiated
    current_layer_args: dict
        All the arguments from model config which are used as kwargs to current_layer_type instance

    Returns
    -------
    keras.layers.Layer
        Instance of current_layer_type
    """
    try:
        return get_keras_layer_subclasses()[current_layer_type](**current_layer_args)
    except KeyError:
        raise KeyError(f"Layer type: '{current_layer_type}' "
                       f"is not supported or not found in subclasses of "
                       f"keras.layers.Layer: '{json.dumps(get_keras_layer_subclasses(), indent=4)}'")
