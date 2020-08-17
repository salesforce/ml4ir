import pandas as pd
import argparse
import logging

#Constanst for the column names given to the query id and graded relevance.
QUERY_ID_NAME = 'qid'
RELEVANCE_NAME = 'relevance'


def process_line(line, keep_additional_info):
    if keep_additional_info == 1:
        feature_values = line.replace('#', '').replace(' = ', ':').strip().split()
    else:
        feature_values = line.split('#')[0].strip().split()
    feature_values[0] = RELEVANCE_NAME + ':' + feature_values[0]
    r = {}
    for fv in feature_values:
        feat = fv.split(':')[0].strip()
        val = fv.split(':')[1].strip()
        if feat == RELEVANCE_NAME or feat == QUERY_ID_NAME:
            r[feat] = val
        else:
            r['f_' + feat] = val
    return r

def convert(input_file, output_file, keep_additional_info, gl_2_clicks):
    f = open(input_file, 'r')
    rows = []

    for line in f:
        rows.append(process_line(line, keep_additional_info))
    f.close()

    df = pd.DataFrame(rows)
    if gl_2_clicks == 1:
        groups = df.groupby(QUERY_ID_NAME)
        for gname, group in groups:
            df.loc[df[QUERY_ID_NAME] == gname, RELEVANCE_NAME] = (
                        df.loc[df[QUERY_ID_NAME] == gname].relevance == max(group.relevance)).astype(int)
    df.to_csv(output_file)


if __name__ == "__main__":
    # Setup logging
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logging.debug("Logger is initialized...")

    # parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='ml4ir/applications/ranking/tests/data/sample.txt', help='ranklib input file path')
    parser.add_argument('--output_file', type=str, default='ml4ir/applications/ranking/tests/data/sample_ml4ir.csv', help='output converted file path')
    parser.add_argument('--keep_additional_info', type=int, default=1,
                        help='Option to keep additional info (All info after the "#") 1 to keep, 0 to ignore')
    parser.add_argument('--gl_2_clicks', type=int, default=1,
                        help='Convert graded relevance to clicks (only max relevant document is considered clicked) 1 to convert')
    args = parser.parse_args()

    convert(args.input_file, args.output_file, args.keep_additional_info, args.gl_2_clicks)
    logging.info('Conversion is completed')
