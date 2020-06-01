# Run from ml4ir/python directory as: python3 ml4ir/applications/ranking/data/scripts/create_dataset.py

import os
import random
import traceback
import datetime as dt
import pandas as pd
import numpy as np
from logging import Logger

from ml4ir.base.io import file_io
from ml4ir.base.io import logging_utils
from ml4ir.base.features.feature_config import parse_config, FeatureConfig


def run_dataset_creation(data_dir: str,
                         out_dir: str,
                         feature_config: str,
                         feature_highval: dict,
                         feature_num_results: str = None,
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

        # Load and parse feature config
        feature_config: FeatureConfig = parse_config(tfrecord_type='', feature_config=feature_config, logger=logger)
        logger.info("Feature config parsed and loaded")

        # Create output location
        file_io.make_directory(out_dir, log=logger)
        out_file = os.path.join(out_dir, 'synthetic_data_{}.csv'.format(dt.datetime.now().strftime('%Y%m%d-%H%M%S')))

        # Build data
        seed_data = load_seed_data(data_dir, logger)

        df_synthetic = fill_data(logger,
                                 seed_data,
                                 max_num_records,
                                 feature_config,
                                 feature_highval,
                                 feature_num_results,
                                 num_samples)
        file_io.write_df(df_synthetic, outfile=out_file, index=False)
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


def load_seed_data(data_dir, logger):
    # Start with loading in CSV format only
    if file_io.path_exists(data_dir):
        dfs = file_io.get_files_in_directory(data_dir)
        return file_io.read_df_list(dfs)
    else:
        logger.error("Error! Data directory must exist and be specified")


def generate_key(seed_dict, query_key, log=None):
    seed_id = str(random.sample(seed_dict[query_key], 1)[0])
    q_id = ''.join(random.sample(seed_id, len(seed_id)))
    if q_id in set(seed_dict.keys()):
        if log:
            log.info('Generated query key is duplicate, regenerating. If happens repeatedly, query key may be too short')
        return generate_key(seed_dict, query_key)
    return q_id


def filter_nres(seed_data, feature_query_key, feature_rank, feature_nres, max_num_records, logger):
    nres = 'num_results_calc'
    if nres in set(seed_data.columns):
        nres = '{}_seed'.format(nres)
    seed_data[nres] = seed_data.groupby(feature_query_key).transform('count')[feature_rank]

    # If the number of results feature name was given, remove queries with not all results included
    if feature_nres and feature_nres in set(seed_data.columns):
        seed_data = seed_data.loc[(seed_data[feature_nres].astype('int64')==seed_data[nres])]
    # Filter to max number of records if given
    if max_num_records:
        seed_data = seed_data.loc[seed_data[nres] <= max_num_records]

    if len(seed_data) == 0:
        logger.error('All seed data has been filtered out!')
    return seed_data, nres


def add_feature_highval(row_dict, feature_highval, is_labeled_pos, q_labelrank):
    # Create ranking-specific high value scenario
    # of the format {high_value_feature_1:[label1_val, label2_val]}
    # Not creating examples of non-catastrophic failure, ie where the clicked record is not a name match
    for hvk in feature_highval.keys():
        if is_labeled_pos and q_labelrank > 1:
            row_dict[hvk] = feature_highval[hvk][1]
        else:
            row_dict[hvk] = feature_highval[hvk][0]
    return row_dict


def fill_data(logger, seed_data, max_num_records, feature_config, feature_highval, feature_nres, num_samples):
    """
    Creates synthetic data using source data as template sampling source
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
    seed_dict_labeled_pos = {}  # In ranking, this is clicked vs unclicked groups
    seed_dict_labeled_neg = {}

    # Get names of essential features from feature_config
    feature_query_key = feature_config.get_query_key('name')
    feature_rank = feature_config.get_rank('name')
    feature_label = feature_config.get_label('name')

    # For ranking data, calculate number of results from data and filter accordingly
    seed_data, nres = filter_nres(seed_data, feature_query_key, feature_rank, feature_nres, max_num_records, logger)

    # For ranking data, calculate clicked rank from seed data as label_rank
    seed_data = pd.merge(seed_data,
                         seed_data[seed_data[feature_label]==True][[feature_rank,feature_query_key]].rename(columns={
                             feature_rank:'label_rank'}),
                         on=feature_query_key,
                         how='left')

    # Now generate data distributions from sample data
    for mycol in list(seed_data.columns):
        seed_dict[mycol] = list(seed_data[mycol].values)
        seed_dict_labeled_pos[mycol] = list(seed_data[seed_data[feature_label] == 1][mycol].values)
        seed_dict_labeled_neg[mycol] = list(seed_data[seed_data[feature_label] == 0][mycol].values)

    for _ in range(num_samples):
        # Generate a synthetic query key
        q_id = generate_key(seed_dict, feature_config.get_query_key('name'))

        # Sample to choose the number of members of the query key (in ranking, the number of results)
        q_nseq = random.sample(seed_dict[nres], 1)[0]

        # Sample to choose which member is labeled positive (in ranking, which result is clicked),
        # sampling only from query keys with the same nseq (number of results)
        q_labelrank = random.sample(list(seed_data[seed_data[nres]==q_nseq].label_rank.values), 1)[0]

        # Build characteristics for each member of the query key (in ranking, each result of the query)
        for i in range(int(q_nseq)):
            is_labeled_pos = (i + 1 == q_labelrank)
            row_dict = {
                feature_query_key: q_id,
                nres: q_nseq,
                'label_rank': q_labelrank,
                feature_rank: i + 1
            }
            for mycol in list(seed_data.columns):
                # Could also let user specify distribution to sample from for each feature
                if mycol not in row_dict.keys() and is_labeled_pos:
                    # row_dict[mycol] = random.sample(seed_dict_labeled_pos[mycol])
                    row_dict[mycol] = None  # Default value, check type # Could also be functions we pass in
                elif mycol not in row_dict.keys() and not is_labeled_pos:
                    # row_dict[mycol] = random.sample(seed_dict_labeled_neg[mycol])
                    row_dict[mycol] = None

            row_dict = add_feature_highval(row_dict, feature_highval, is_labeled_pos, q_labelrank)

            rows_df.append(row_dict)
    return pd.DataFrame(rows_df)


def main():
    data_dir = 'ml4ir/applications/ranking/data/example/'
    out_dir = 'ml4ir/applications/ranking/data/synthetic/'
    feature_config = 'ml4ir/applications/ranking/data/example/feature_config.yaml'
    feature_highval = {'name_match':[0,1]}
    feature_num_results = 'num_results'
    max_num_records = 50
    num_samples = 10
    random_state = 123
    logger = None

    run_dataset_creation(data_dir,
                         out_dir,
                         feature_config,
                         feature_highval,
                         feature_num_results,
                         max_num_records,
                         num_samples,
                         random_state,
                         logger)


if __name__ == "__main__":
    main()
