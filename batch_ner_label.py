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
    # print(f'{input_path}/{file_name}')
    # print(df.shape)
    # print(df.columns)
    tags = df.apply(get_tags, axis=1)
    tags.columns = ['ent0', 'typ0', 'ent1', 'typ1', 'ent2', 'typ2']
    # print(df.columns)
    df = df.drop('results', axis=1)
    # print(df.columns)
    df = pd.concat([df, tags], axis=1)
    # print(df.columns)
    df.to_parquet(f'{output_path}/{file_name}')


def set_argparsing():
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--source', type=str, nargs='+',required=True)
    parser.add_argument('--stage', type=str, default="tags") # 
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    args = set_argparsing()
    DATA_NAME_LIST = args.source
    stage=args.stage
    
    # input_path = f'{DATA_NAME}/tags'
    # output_path = f'{DATA_NAME}/final'
    for DATA_NAME in DATA_NAME_LIST:
        print(f"starting processing {DATA_NAME}...")
        input_path=f'{DATA_NAME}/{stage}'
        output_path=f'{DATA_NAME}/final'
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        for file in tqdm(glob.glob(f'{input_path}/*.parquet')):
            process(file.split('/')[-1], input_path, output_path)
            # break
        print(f"Finished {DATA_NAME}")
