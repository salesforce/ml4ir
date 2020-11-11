import ast
from typing import List, Dict


DOT = "."
NAME = "name"


def cast_dynamic_val(val: str):
    """
    Cast the dynamic val from string to python base data type

    Parameters
    ----------
    val: str
        String value to be cast

    Returns
    -------
    Value cast to appropriate dtype
    """
    if isinstance(val, str):
        if val.replace(DOT, "").isdigit():
            if DOT in val:
                # float
                return float(val)
            else:
                # int
                return int(val)
        elif val.isalnum():
            # str
            return val
        else:
            # If contains punctuation
            return ast.literal_eval(val)
    else:
        return val


def override_list(base_list: List, dynamic_key: str, val):
    """
    Customize the base list by updating with the
    dynamic_key and val.

    Parameters
    ----------
    base: dict
        Dictionary or List to be customized with dynamic args
    dynamic_key: str
        Key to identify the location the value should be updated.
        Nested with DOT like "custom.key_0.key_1.key_2.0.0.key_4"
    val: str or float or int or dict or list
        Value to be set

    Returns
    -------
    dict
        Updated base_list based on the key-value pairs in dynamic_args

    Notes
    -----
    This will be called recursively with override_dict.
    If dynamic_key is not a number, then we try to match on `name` field
    in the list of dictionaries.
    """
    def find_root_key_index(base_list, root_key):
        if root_key.isdigit():
            # If array index
            root_key = int(root_key)
        else:
            # If string, then match on `name`
            for root_key_i in range(len(base_list)):
                if root_key == base_list[root_key_i][NAME]:
                    root_key = root_key_i
                    break

            if not isinstance(root_key, int):
                raise KeyError("{} not found in List".format(root_key))

        return root_key

    if DOT in dynamic_key:
        # Compute root and subtree keys
        root_key = find_root_key_index(base_list, dynamic_key.split(DOT)[0])
        subtree_key = DOT.join(dynamic_key.split(DOT)[1:])

        # Extract subtree
        subtree = base_list[root_key]

        if isinstance(subtree, dict):
            root_val = override_dict(base_dict=subtree,
                                     dynamic_key=subtree_key,
                                     val=val)
        elif isinstance(subtree, list):
            root_val = override_list(base_list=subtree,
                                     dynamic_key=subtree_key,
                                     val=val)
        else:
            raise ValueError(
                "Unsupported subtree type. Must be one of list or dict")
    else:
        # End of nested dynamic key
        root_key = find_root_key_index(base_list, dynamic_key)
        root_val = val

    base_list[root_key] = root_val

    return base_list


def override_dict(base_dict: Dict, dynamic_key: str, val):
    """
    Customize the base dictionary by updating with the
    dynamic_key and val.

    Parameters
    ----------
    base: dict
        Dictionary or List to be customized with dynamic args
    dynamic_key: str
        Key to identify the location the value should be updated.
        Nested with DOT like "custom.key_0.key_1.key_2.0.0.key_4"
    val: str or float or int or dict or list
        Value to be set

    Returns
    -------
    dict
        Updated base_dict based on the key-value pairs in dynamic_args

    Notes
    -----
    This will be called recursively along with override_list
    """
    if DOT in dynamic_key:
        # Compute root and subtree keys
        root_key = dynamic_key.split(DOT)[0]
        subtree_key = DOT.join(dynamic_key.split(DOT)[1:])

        # Extract subtree
        subtree = base_dict[root_key]

        if isinstance(subtree, dict):
            root_val = override_dict(base_dict=subtree,
                                     dynamic_key=subtree_key,
                                     val=val)
        elif isinstance(subtree, list):
            root_val = override_list(base_list=subtree,
                                     dynamic_key=subtree_key,
                                     val=val)
        else:
            raise ValueError(
                "Unsupported subtree type. Must be one of list or dict")
    else:
        # End of nested dynamic key
        root_key = dynamic_key
        root_val = val

    base_dict[root_key] = root_val

    return base_dict


def override_with_dynamic_args(base_dict: Dict, dynamic_args: Dict = {}):
    """
    Customize the base dictionary by updating with the
    key-value pairs in dynamic args.

    Parameters
    ----------
    base_dict: dict
        Dictionary to be customized with dynamic args
    dynamic_args: dict
        Flat dictionary with keys and values used to update the base_dict

    Returns
    -------
    dict
        Updated base_dict based on the key-value pairs in dynamic_args

    Notes
    -----
    This is used to override
    base feature_config and model_config presets from command line arguments.
    """
    for key, val in dynamic_args.items():
        base_dict = override_dict(base_dict=base_dict,
                                  dynamic_key=key,
                                  val=cast_dynamic_val(val))

    return base_dict
