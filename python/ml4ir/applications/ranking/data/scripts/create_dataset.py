# Run from ml4ir/python directory as: python3 ml4ir/applications/ranking/data/scripts/create_dataset.py

import argparse
import os
import random
import socket
import time
import traceback
import datetime as dt
import pandas as pd
import numpy as np
from logging import Logger

from ml4ir.base.io.local_io import LocalIO
from ml4ir.base.io import logging_utils
from ml4ir.base.features.feature_config import FeatureConfig
from ml4ir.base.config.keys import TFRecordTypeKey

# Defaults
DATA_DIR = "ml4ir/applications/ranking/data/"
OUT_DIR = "ml4ir/applications/ranking/data/synthetic/"
FEATURE_CONFIG = "ml4ir/applications/ranking/data/feature_config.yaml"
FEATURE_HIGHVAL = {"text_match_bool": [0, 1]}
FEATURE_NUM_RESULTS = "num_results"
MAX_NUM_RECORDS = 50
NUM_SAMPLES = 10
RANDOM_STATE = 123


def run_dataset_creation(
    data_dir: str = DATA_DIR,
    out_dir: str = OUT_DIR,
    feature_config_path: str = FEATURE_CONFIG,
    feature_highval: dict = FEATURE_HIGHVAL,
    feature_num_results: str = FEATURE_NUM_RESULTS,
    max_num_records: int = MAX_NUM_RECORDS,
    num_samples: int = NUM_SAMPLES,
    random_state: int = RANDOM_STATE,
):
    """
    1. Loads example data
    2. Builds specified synthetic data size by sampling from example data
    3. Adds catastrophic failures specifically
    4. For now, write out to CSV. In future could return df directly
    """
    # Setup logging
    file_io = LocalIO()
    logger: Logger = setup_logging(file_io)
    file_io.set_logger(logger)

    try:
        # Set seeds
        set_seeds(random_state)
        logger.info("Set seeds with initial random state {}".format(random_state))

        # Load and parse feature config
        feature_config: FeatureConfig = FeatureConfig.get_instance(
            tfrecord_type=TFRecordTypeKey.SEQUENCE_EXAMPLE,
            feature_config_dict=file_io.read_yaml(feature_config_path),
            logger=logger,
        )
        logger.info("Feature config parsed and loaded")

        # Create output location
        file_io.make_directory(out_dir)
        out_file = os.path.join(
            out_dir, "synthetic_data_{}.csv".format(dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
        )

        # Build data
        seed_data = load_seed_data(data_dir, logger, file_io)

        df_synthetic = fill_data(
            seed_data,
            max_num_records,
            feature_config,
            feature_highval,
            feature_num_results,
            num_samples,
            logger,
        )
        file_io.write_df(df_synthetic, outfile=out_file, index=False)
        logger.info("Synthetic data created! Location: {}".format(out_file))
        return df_synthetic

    except Exception as e:
        logger.error("!!! Error creating synthetic data: !!!\n{}".format(str(e)))
        traceback.print_exc()
        return


def setup_logging(file_io: LocalIO):
    run_id = "-".join([socket.gethostname(), time.strftime("%Y%m%d-%H%M%S")])
    logs_dir: str = os.path.join("logs", run_id)
    file_io.make_directory(logs_dir, clear_dir=True)

    outfile: str = os.path.join(logs_dir, "output_log.csv")
    logger = logging_utils.setup_logging(reset=True, file_name=outfile, log_to_file=True)

    logger.info("Logging initialized. Saving logs to : {}".format(logs_dir))
    logger.info("Run ID: {}".format(run_id))
    return logger


def set_seeds(random_state):
    # for repeatability
    np.random.seed(random_state)
    random.seed(random_state)


def load_seed_data(data_dir, logger, file_io: LocalIO):
    # Start with loading in CSV format only
    if file_io.path_exists(data_dir):
        dfs = file_io.get_files_in_directory(data_dir)
        return file_io.read_df_list(dfs)
    else:
        logger.error("Error! Data directory must exist and be specified")


def generate_key(seed_dict, query_key, logger):
    seed_id = str(random.sample(seed_dict[query_key], 1)[0])
    q_id = "".join(random.sample(seed_id, len(seed_id)))
    if q_id in set(seed_dict.keys()):
        logger.warning(
            "Generated query key is duplicate, regenerating. If happens repeatedly, query key may be too short"
        )
        return generate_key(seed_dict, query_key, logger)
    return q_id


def filter_nres(seed_data, feature_query_key, feature_rank, feature_nres, max_num_records, logger):
    nres = "num_results_calc"
    if nres in set(seed_data.columns):
        nres = "{}_seed".format(nres)
    seed_data[nres] = seed_data.groupby(feature_query_key).transform("count")[feature_rank]

    # If the number of results feature name was given, remove queries with not all results included
    if feature_nres and feature_nres in set(seed_data.columns):
        seed_data = seed_data.loc[
            (seed_data[feature_nres].dropna().astype("int64") == seed_data[nres])
        ]
    # Filter to max number of records if given
    if max_num_records:
        seed_data = seed_data.loc[seed_data[nres] <= max_num_records]

    if len(seed_data) == 0:
        logger.error("All seed data has been filtered out!")
    return seed_data, nres


def add_feature_highval(row_dict, feature_highval, is_clicked, click_rank):
    # Create ranking-specific high value scenario
    # of the format {high_value_feature_1:[label1_val, label2_val]}
    # Not creating examples of non-catastrophic failure, ie where the clicked record is not a name match
    for hvk in feature_highval.keys():
        if is_clicked and click_rank > 1:
            row_dict[hvk] = feature_highval[hvk][1]
        else:
            row_dict[hvk] = feature_highval[hvk][0]
    return row_dict


def fill_data(
    seed_data, max_num_records, feature_config, feature_highval, feature_nres, num_samples, logger
):
    """
    Creates synthetic data using source data as template sampling source
    Fastest way is to create a list of dicts, then create the DF from the list
    For the specified number of samples:
    1. Generate a unique query key
    2. Random sample (with replacement) number of results from seed data
    3. Random sample (with replacement) to fill each result from seed data
    """

    # Set up new DF and seed data distributions for sampling
    rows_df = []
    seed_dict = (
        {}
    )  # the keys are seed data features, the values are lists of associated values in seed data
    seed_dict_clicked = {}
    seed_dict_unclicked = {}

    # Get names of essential features from feature_config
    feature_query_key = feature_config.get_query_key("name")
    feature_rank = feature_config.get_rank("name")
    feature_label = feature_config.get_label("name")

    # For ranking data, calculate number of results from data and filter accordingly
    seed_data, nres = filter_nres(
        seed_data, feature_query_key, feature_rank, feature_nres, max_num_records, logger
    )

    # For ranking data, get rank of clicked result
    seed_data = pd.merge(
        seed_data,
        seed_data[seed_data[feature_label] == True][
            [feature_rank, feature_query_key]
        ].rename(  # noqa: E712 # FIXME
            columns={feature_rank: "click_rank"}
        ),
        on=feature_query_key,
        how="left",
    )

    # Now generate data distributions from seed data
    for mycol in list(seed_data.columns):
        seed_dict[mycol] = list(seed_data[mycol].values)
        seed_dict_clicked[mycol] = list(seed_data[seed_data[feature_label] == 1][mycol].values)
        seed_dict_unclicked[mycol] = list(seed_data[seed_data[feature_label] == 0][mycol].values)

    for _ in range(num_samples):
        # Generate a synthetic query key
        q_id = generate_key(seed_dict, feature_query_key, logger)

        # Sample to choose the number of members of the query key (in ranking, the number of results)
        q_nseq = random.sample(seed_dict[nres], 1)[0]

        # Sample to choose which result is clicked, sampling only from query keys with the same nseq (number of results)
        click_rank = random.sample(
            list(seed_data[seed_data[nres] == q_nseq].click_rank.values), 1
        )[0]

        # Build characteristics for each member of the query key (in ranking, each result of the query)
        for i in range(int(q_nseq)):
            is_clicked = i + 1 == click_rank
            row_dict = {
                feature_query_key: q_id,
                nres: q_nseq,
                "click_rank": click_rank,
                feature_rank: i + 1,
                feature_label: int(is_clicked),
            }
            for mycol in list(seed_data.columns):
                # Could also let user specify distribution to sample from for each feature
                if mycol not in row_dict.keys() and is_clicked:
                    # row_dict[mycol] = random.sample(seed_dict_clicked[mycol])
                    row_dict[
                        mycol
                    ] = None  # Default value, check type # Could also be functions we pass in
                elif mycol not in row_dict.keys() and not is_clicked:
                    # row_dict[mycol] = random.sample(seed_dict_unclicked[mycol])
                    row_dict[mycol] = None

            row_dict = add_feature_highval(row_dict, feature_highval, is_clicked, click_rank)

            rows_df.append(row_dict)
    return pd.DataFrame(rows_df)


def main(args):

    run_dataset_creation(
        args.data_dir,
        args.out_dir,
        args.feature_config,
        args.feature_highval,
        args.feature_num_results,
        args.max_num_records,
        args.num_samples,
        args.random_state,
    )


if __name__ == "__main__":
    # Define args
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_dir", default=DATA_DIR)
    parser.add_argument("-out_dir", default=OUT_DIR)
    parser.add_argument("-feature_config", default=FEATURE_CONFIG)
    parser.add_argument("-feature_highval", default=FEATURE_HIGHVAL)
    parser.add_argument("-feature_num_results", default=FEATURE_NUM_RESULTS)
    parser.add_argument("-max_num_records", default=MAX_NUM_RECORDS)
    parser.add_argument("-num_samples", default=NUM_SAMPLES)
    parser.add_argument("-random_state", default=RANDOM_STATE)
    args = parser.parse_args()
    main(args)
