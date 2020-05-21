# Run from ml4ir/python/applications/ranking directory
# Run from ml4ir/python directory as: python3 applications/ranking/scripts/run_create_dataset.py

from applications.ranking.data.scripts.create_dataset import create_dataset


def main():
    data_dir = 'applications/ranking/data/example/'
    data_out_dir = 'applications/ranking/data/synthetic/'
    feature_config_src = 'applications/ranking/data/example/feature_config.yaml'
    max_num_records = 50
    num_samples = 10
    random_state = 123
    logger = None

    create_dataset(data_dir, data_out_dir, feature_config_src, max_num_records, num_samples, random_state, logger)


if __name__ == "__main__":
    main()
