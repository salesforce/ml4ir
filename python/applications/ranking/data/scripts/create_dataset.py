import os
import random
import traceback
import datetime
import pandas as pd
import numpy as np
from logging import Logger

from ml4ir.io import file_io
from ml4ir.io import logging_utils
from ml4ir.features.feature_config import parse_config, FeatureConfig


def create_dataset(data_dir: str,
                   data_out_dir: str,
                   feature_config_src: str,
                   max_num_records: int = 5,
                   num_samples: int = 10,
                   random_state: int = 123,
                   logger: Logger = None):
    """
    1. Loads example data
    2. Builds specified synthetic data size by sampling from example data
    3. Adds catastrophic failures specifically
    4. For now, write out to CSV. In future could return df directly
    """

    # Setup logging
    logs_dir: str = os.path.join('logs', 'test_run_id')
    file_io.make_directory(logs_dir, clear_dir=True, log=None)
    logger: Logger = setup_logging()
    logger.info('Logging initialized. Saving logs to : {}'.format(logs_dir))
    logger.info('Run ID: {}'.format('test_run_id'))

    try:
        # Set seeds
        set_seeds(random_state)
        logger.info('Set seeds with initial random state {}'.format(random_state))

        # Load and parse feature config #TODO check on tfrecord_type
        feature_config: FeatureConfig = parse_config(tfrecord_type='', feature_config=feature_config_src, logger=logger)
        logger.info("Feature config parsed and loaded")

        # Create output location
        if not os.path.exists(data_out_dir):
            os.makedirs(data_out_dir)
        out_file = os.path.join(data_out_dir,
                                'synthetic_data_{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S')))

        # Build data
        example_data = load_example_data(data_dir, logger)

        df_synthetic = fill_data(logger, example_data, max_num_records, feature_config, num_samples)
        df_synthetic.to_csv(out_file, index=False)
        logger.info('Synthetic data created! Location: {}'.format(out_file))
        return df_synthetic

    except Exception as e:
        logger.error("!!! Error creating synthetic data: !!!\n{}".format(str(e)))
        traceback.print_exc()
        return


def setup_logging():
    return logging_utils.setup_logging(reset=True, file_name='test.txt', log_to_file=True)


def set_seeds(random_state):
    # for repeatability
    np.random.seed(random_state)
    random.seed(random_state)


def load_example_data(data_dir, logger):
    # Start with loading in CSV format only
    if file_io.path_exists(data_dir):
        dfs = file_io.get_files_in_directory(data_dir)
        return file_io.read_df_list(dfs)
    else:
        logger.error("Error! Data directory must exist and be specified")


def fill_data(logger, example_data, max_num_records, feature_config, num_samples):
    """
    Creates synthetic data using source data as template/optional sampling source
    Fastest way is to create a list of dicts, then create the DF from the list
    For the specified number of samples:
    1. Generate a query key
    2. Random sample number of results (with replacement)
    3. Random sample to fill each result (using FeatureConfig)
    4. TODO Check that no record ID is used twice
    """

    # Set up new DF and example data distributions for sampling options
    rows_df = []
    seed_dict = {}  # the keys are source data columns, the values are lists of source data column values
    seed_dict_labeled_pos = {}  # option to get characteristics of labeled results separately
    seed_dict_labeled_neg = {}

    name_num_sequences = [x['serving_info']['name'] for x in feature_config.all_features if x['name']=='num_sequences'][0]
    name_label_rank = [x['serving_info']['name'] for x in feature_config.all_features if x['name']=='label_rank'][0]
    name_highval_feature = [x['serving_info']['name'] for x in feature_config.all_features if x['name']=='high_val_feature_1'][0]

    for mycol in list(example_data.columns):
        # Generate data distributions from sample data
        seed_dict[mycol] = list(example_data[mycol].values)
        seed_dict_labeled_pos[mycol] = list(example_data[example_data[feature_config.get_label('name')] == 1].values)
        seed_dict_labeled_neg[mycol] = list(example_data[example_data[feature_config.get_label('name')] == 0].values)

    seed_dict[name_num_sequences] = list(filter(lambda x: x <= max_num_records, seed_dict[name_num_sequences]))

    for _ in range(num_samples):
        # Generate a synthetic query ID
        seed_id = str(random.sample(seed_dict[feature_config.get_query_key('name')], 1)[0])
        q_id = ''.join(random.sample(seed_id, len(seed_id)))

        # Sample the number of results and label rank from example data
        q_nseq = random.sample(seed_dict[name_num_sequences], 1)[0]
        q_labelrank = random.sample(seed_dict[name_label_rank], 1)[0]

        for i in range(int(q_nseq)):
            is_labeled_pos = (i + 1 == q_labelrank)
            row_dict = {
                feature_config.get_query_key('name'): q_id,
                name_num_sequences: q_nseq,
                name_label_rank: q_labelrank,
                feature_config.get_rank('name'): i + 1
            }
            for mycol in list(example_data.columns):
                # Could also let user specify distribution to sample from for each feature
                if mycol not in row_dict.keys() and is_labeled_pos:
                    # row_dict[mycol] = random.sample(seed_dict_labeled_pos[mycol])
                    row_dict[mycol] = None  # Default value, check type # Could also be functions we pass in
                elif mycol not in row_dict.keys() and not is_labeled_pos:
                    # row_dict[mycol] = random.sample(seed_dict_labeled_neg[mycol])
                    row_dict[mycol] = None

            # Create catastrophic failure # Rename, possible to extenuate this for each high value feature
            # {high_value_feature_1:[label1_val, label2_val], high_value_feature_2: [unclicked_val, clicked_val]}
            # Not creating examples of non-catastrophic failure, ie where the clicked record is not a name match
            # Configurable feature to pass in here! Pass in: feature name
            if is_labeled_pos and q_labelrank > 1:
                row_dict[name_highval_feature] = 1
            else:
                row_dict[name_highval_feature] = 0

            rows_df.append(row_dict)
    return pd.DataFrame(rows_df)
