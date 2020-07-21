from random import seed
from random import randint

import csv


CSV_TRAIN_FILE_PATH = "data/csv/train/file_0.csv"
CSV_TEST_FILE_PATH = "data/csv/test/file_0.csv"
CSV_VALIDATION_FILE_PATH = "data/csv/validation/file_0.csv"
COLUMNS_HEADER = ["query_key", "query_feature_0", "group_feature_1", "group_sequence_feature_2", "entity"]
QUERY_VOCABULARY = [
    "the", "tragedy", "of", "hamlet", "prince", "denmark", "shakespeare", "homepage", "entire", "play", "act", "i",
    "scene", "elsinore", "a", "platform", "before", "castle", "francisco", "at", "his", "post", "enter", "to", "him",
    "bernardo", "whos", "there", "nay", "answer", "me", "stand", "and", "unfold", "yourself", "long", "live", "king",
    "he", "you", "come", "most", "carefully", "upon", "your", "hour", "tis", "now", "struck", "twelve", "get", "thee",
    "bed", "for", "this", "relief", "much", "thanks", "bitter", "cold", "am", "sick", "heart", "have", "had", "quiet",
    "guard", "not", "mouse", "stirring", "well", "good", "night", "if", "do", "meet", "horatio", "marcellus", "rivals",
    "my", "watch", "bid", "them", "make", "haste", "think", "hear", "ho", "friends", "ground", "liegemen", "dane"
]

FEATURE_1_VOCABULARY = [str(i) for i in range(0, 20)]

FEATURE_2_VOCABULARY = [
    "AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH"
]

LABEL_VOCABULARY = [str(i) for i in range(0, 10)]


class FeatureGenerator:

    def __init__(self, max_token_length, vocabulary, sequence_joiner=" "):
        self.__max_token_length = max_token_length
        self.__vocabulary = vocabulary
        self.__vocabulary_length = len(vocabulary) - 1
        self.__sequence_joiner = sequence_joiner

    def generate_feature(self):
        query_tokens = [self.__vocabulary[randint(0, self.__vocabulary_length)]
                        for _ in range(0, randint(1, self.__max_token_length))]
        return self.__sequence_joiner.join(query_tokens)


def generate_csv_test_data():
    seed(123)
    query_feature_0_generator = FeatureGenerator(7, QUERY_VOCABULARY)
    group_feature_1_generator = FeatureGenerator(1, FEATURE_1_VOCABULARY)
    group_sequence_feature_2_generator = FeatureGenerator(20, FEATURE_2_VOCABULARY, sequence_joiner=",")
    label_generator = FeatureGenerator(1, LABEL_VOCABULARY)
    generators = [
        query_feature_0_generator,
        group_feature_1_generator,
        group_sequence_feature_2_generator,
        label_generator
    ]

    for (path, number_rows) in [(CSV_TRAIN_FILE_PATH, 7000), (CSV_TEST_FILE_PATH, 2000),
                                (CSV_VALIDATION_FILE_PATH, 1000)]:
        rows = [COLUMNS_HEADER] + [['query_id_' + str(idx)] + [g.generate_feature() for g in generators] for idx in range(0, number_rows)]
        with open(path, 'w', newline='') as file:
            writer = csv.writer(file, quoting=csv.QUOTE_NONNUMERIC)
            writer.writerows(rows)


def main():
    generate_csv_test_data()
    return


if __name__ == "__main__":
    main()