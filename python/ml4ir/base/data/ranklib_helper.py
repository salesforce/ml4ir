import ast
import argparse
import pandas as pd
import numpy as np

max_f_id = 0
def process_line(line, keep_additional_info, query_id_name, relevance_name):
    """Takes an input line in ranklib format and returns a row in ml4ir format.

        Parameters
        ----------
        line : str
            a line from ranklib format data
        keep_additional_info : bool
            Option to keep additional info (All info after the "#") True to keep, False to ignore.
        query_id_name : str
            The name of the query id column.
        relevance_name : str
            The name of the relevance column (the target label).


        Returns
        -------
        dictionary <column:value>
            Keys are the column names values are the parsed values.
    """

    if keep_additional_info:
        feature_values = line.replace('#', '').replace(' = ', ':').strip().split()
    else:
        feature_values = line.split('#')[0].strip().split()
    feature_values[0] = relevance_name + ':' + feature_values[0]
    r = {}
    for fv in feature_values:
        feat = fv.split(':')[0].strip()
        if feat == query_id_name:
            val = fv.split(':')[1].strip()
            r[feat] = 'Q'+str(val)
        elif feat == relevance_name:
            val = fv.split(':')[1].strip()
            r[feat] = float(val)
        else:
            try:
                val = float(fv.split(':')[1].strip())
            except:
                val = fv.split(':')[1].strip()
            r['f_' + feat] = val
            global max_f_id
            if int(feat) > max_f_id:
                max_f_id = int(feat)
    return r


def convert(input_file, keep_additional_info, gl_2_clicks,
            non_zero_features_only, query_id_name, relevance_name,
            add_dummy_rank_column = True):
    """Convert the input file with the specified parameters into ml4ir format. returns a dataframe

            Parameters
            ----------
            input_file : str
                ranklib input file path
            keep_additional_info : bool
                Option to keep additional info (All info after the "#") True to keep, False to ignore.
            gl_2_clicks : int
                Convert graded relevance to clicks (only max relevant document is considered clicked) 1 to convert
            query_id_name : str
                The name of the query id column.
            relevance_name : str
                The name of the relevance column (the target label).
            add_dummy_rank_column : bool
                ml4ir expects pre-rankings. This would add a dummy pre-rankings.


            Returns
            -------
            Dataframe
                converted ml4ir dataframe
        """

    f = open(input_file, 'r')
    rows = []
    for line in f:
        rows.append(process_line(line, keep_additional_info, query_id_name, relevance_name))
    f.close()
    if non_zero_features_only:
        columns = [query_id_name, relevance_name] + ['f_' + str(i) for i in range(max_f_id)]
        df = pd.DataFrame(rows, columns=columns)
        df.replace(np.nan, 0, inplace=True)
    else:
        df = pd.DataFrame(rows)
    if int(gl_2_clicks) == 1:
        groups = df.groupby(query_id_name)
        for gname, group in groups:
            df.loc[df[query_id_name] == gname, relevance_name] = (
                    df.loc[df[query_id_name] == gname].relevance == max(group.relevance)).astype(int)

    # NOTE: ml4ir expects a pre-ranking. Adding a dummy pre-ranking to match format.
    if add_dummy_rank_column:
        df['rank'] = 1
    return df

def ranklib_to_csv(input_file, output_file, keep_additional_info, gl_2_clicks, non_zero_features_only, query_id_name, relevance_name, add_dummy_rank_column = False):
    """Convert the input file with the specified parameters into ml4ir format writes the converted file to a csv

            Parameters
            ----------
            input_file : str
                ranklib input file path
            output_file : str
                output converted file path
            keep_additional_info : bool
                Option to keep additional info (All info after `#`) True to keep, False to ignore.
            gl_2_clicks : int
                Convert graded relevance to clicks (only max relevant document is considered clicked) 1 to convert
            query_id_name : str
                The name of the query id column.
            relevance_name : str
                The name of the relevance column (the target label).
            add_dummy_rank_column : bool
                ml4ir expects pre-rankings. This would add a dummy pre-rankings.
    """

    df = convert(input_file, keep_additional_info, gl_2_clicks, non_zero_features_only, query_id_name, relevance_name, add_dummy_rank_column)
    df.to_csv(output_file)

def ranklib_directory_to_csvs(input_dir, keep_additional_info, gl_2_clicks, non_zero_features_only, query_id_name, relevance_name, add_dummy_rank_column = False):
    """Convert all files in the given directory with the specified parameters into ml4ir format writes the converted file to a csv
            Parameters
            ----------
            input_dir : str
                ranklib input directory path. All files within the directory will be converted.
            keep_additional_info : bool
                Option to keep additional info (All info after `#`) True to keep, False to ignore.
            gl_2_clicks : int
                Convert graded relevance to clicks (only max relevant document is considered clicked) 1 to convert
            query_id_name : str
                The name of the query id column.
            relevance_name : str
                The name of the relevance column (the target label).
            add_dummy_rank_column : bool
                ml4ir expects pre-rankings. This would add a dummy pre-rankings.
    """
    from os import listdir
    from os.path import isfile, join
    onlyfiles = [f for f in listdir(input_dir) if isfile(join(input_dir, f))]
    for f in onlyfiles[1:]:
        ranklib_to_csv(join(input_dir, f), join(input_dir, f)+'_ml4ir.csv', keep_additional_info, gl_2_clicks, non_zero_features_only, query_id_name, relevance_name, add_dummy_rank_column)



if __name__ == "__main__":
    # parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='ml4ir/applications/ranking/tests/data/train/sample.txt', help='ranklib input file path')
    parser.add_argument('--input_dir', type=str, default='ml4ir/applications/ranking/tests/data/test',
                        help='ranklib input directory path. All files within the directory will be converted.')
    parser.add_argument('--output_file', type=str, default='ml4ir/applications/ranking/tests/data/train/sample_ml4ir.csv', help='output converted file path')
    parser.add_argument('--keep_additional_info', type=ast.literal_eval, default=True,
                        help='Option to keep additional info (All info after the "#") True to keep, False to ignore')
    parser.add_argument('--gl_2_clicks', type=int, default=1,
                        help='Convert graded relevance to clicks (only max relevant document is considered clicked) 1 to convert')
    parser.add_argument('--non_zero_features_only', type=int, default=True,
                        help='Only non zero features are stored. True for yes, False otherwise')
    parser.add_argument('--query_id_name', type=str, default='qid',
                        help='The name of the query id column.')
    parser.add_argument('--relevance_name', type=str, default='relevance',
                        help='The name of the relevance column.')

    args = parser.parse_args()
    print("Converting file...")
    ranklib_to_csv(args.input_file, args.output_file, args.keep_additional_info, args.gl_2_clicks,
                   args.non_zero_features_only, args.query_id_name, args.relevance_name)
    print('Conversion is completed')
