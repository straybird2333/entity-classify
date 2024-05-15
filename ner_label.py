import os
import glob
import argparse
import pandas as pd
from tqdm import tqdm
from collections import Counter

SELECT_NUM = 3

def get_tags(arr):
    counter = eval(arr['results'])
    tags = [None, None] * SELECT_NUM
    for idx, tag in enumerate(counter.most_common(SELECT_NUM)):
        if tag[1] <= 1: continue
        tags[idx * 2] = tag[0][0]
        tags[idx *2 + 1] = tag[0][1]
    return pd.Series(tags)

def process(file_name, input_path, output_path):
    df = pd.read_parquet(f'{input_path}/{file_name}', engine='pyarrow')
    # print(df.shape)

    tags = df.apply(get_tags, axis=1)
    tags.columns = ['ent0', 'typ0', 'ent1', 'typ1', 'ent2', 'typ2']

    df = df.drop('results', axis=1)
    df = pd.concat([df, tags], axis=1)
    df.to_parquet(f'{output_path}/{file_name}')


def set_argparsing():
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--source', type=str,required=True)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    args = set_argparsing()
    DATA_NAME = args.source
    
    # input_path = f'{DATA_NAME}/tags'
    # output_path = f'{DATA_NAME}/final'
    
    input_path=f'{DATA_NAME}/tags'
    output_path=f'{DATA_NAME}/final'
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for file in tqdm(glob.glob(f'{input_path}/*.parquet')):
        process(file.split('/')[-1], input_path, output_path)
