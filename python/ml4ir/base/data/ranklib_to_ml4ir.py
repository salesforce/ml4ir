import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--inputfilePath', type=str, default='', help='ranklib input file path')
parser.add_argument('--outputfilePath', type=str, default='', help='output converted file path')
parser.add_argument('--keepAdditionalInfo', type=int, default=1, help='Option to keep additional info (All info after the "#") 1 to keep, 0 to ignore')
parser.add_argument('--GL2Clicks', type=int, default=1, help='Convert graded relevance to clicks (only max relevant document is considered clicked) 1 to convert')
args = parser.parse_args()
query_id_name = 'qid'
relevace_name = 'relevance'

f = open(args.inputfilePath, 'r')
cntr = 0
rows = []
for line in f:
    if args.keepAdditionalInfo == 1:
        feature_values = line.replace('#','').replace(' = ', ':').strip().split()
    else:
        feature_values = line.split('#')[0].strip().split()
    feature_values[0] = relevace_name+':'+feature_values[0]
    r={}
    for fv in feature_values:
        feat = fv.split(':')[0].strip()
        val = fv.split(':')[1].strip()
        if feat == relevace_name or feat == query_id_name:
            r[feat] = val
        else:
            r['f_'+feat] = val
    rows.append(r)
        #r = {'f_'+fv.split(':')[0].strip() : fv.split(':')[1].strip() for fv in feature_values}

f.close()
df = pd.DataFrame(rows)
if args.GL2Clicks == 1:
    groups = df.groupby(query_id_name)
    for gname, group in groups:
        df.loc[df[query_id_name] == gname, relevace_name] = (df.loc[df[query_id_name] == gname].relevance == max(group.relevance)).astype(int)
df.to_csv(args.outputfilePath)
print('DONE!')


