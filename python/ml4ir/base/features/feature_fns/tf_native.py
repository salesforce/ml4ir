import tensorflow as tf

from ml4ir.base.io.file_io import FileIO


def tf_native_op(feature_tensor: tf.Tensor, feature_info: dict, file_io: FileIO):
    """
    Run a series of tensorflow native operations on the input feature tensor.
    The functions will be applied in the order they are specified.

    Parameters
    ----------
    feature_tensor : Tensor
        Input feature tensor
    feature_info : dict
        Dictionary representing the configuration parameters for the specific feature from the FeatureConfig
    file_io : FileIO object
        FileIO handler object for reading and writing

    Returns
    -------
    Tensor object
        Modified feature tensor after applying all the specified ops

    Notes
    -----
    Args under feature_layer_info:
        ops: list of dict
            List of function specifications with associated arguments
            
            Arguments under opts:
                fn : str
                    Tensorflow native function name. Should start with tf.
                    Example: tf.math.log or tf.clip_by_value
                args : dict
                    Keyword arguments to be passed to the tensorflow function
    """
    feature_node_name = feature_info.get("node_name", feature_info.get("name"))
    tf_ops = feature_info.get("feature_layer_info", {}).get("args", {}).get("ops", {})

    if not tf_ops:
        return feature_tensor

    for tf_op in tf_ops:
        try:
            fn_, fn_args = eval(tf_op["fn"]), tf_op.get("args", {})
        except AttributeError as e:
            raise KeyError("Invalid fn specified for tf_native_op : {}\n{}".format(tf_op["fn"], e))
        
        try:
            feature_tensor = fn_(feature_tensor, **fn_args)
        except Exception as e:
            raise Exception("Error while applying {} to {} feature:\n{}".format(tf_op["fn"], feature_node_name, e))

    # Adjusting the shape to the default feature fns for concatenating in the next step
    feature_tensor = tf.expand_dims(feature_tensor, axis=-1, name="{}_tf_native_op".format(feature_node_name))

    return feature_tensor
