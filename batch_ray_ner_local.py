import ray
import argparse
# import dask.dataframe as da
import pandas as pd
from paddlenlp import Taskflow
from collections import Counter
import json, os, pickle
from tqdm import tqdm
import time
import glob
import logging
from collections import Counter
import pyarrow.parquet as pq
import pyarrow as pa


BATCH_SIZE = 500 # 666 error? 700 error
DATA_NAME = None


def set_logging(): # 设置logger
    logging.basicConfig(level = logging.DEBUG,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s', filename='logging.txt')
    logger = logging.getLogger(__name__)
    # logger.info("Start print log") logger.debug("Do something") logger.warning("Something maybe fail.") logger.info("Finish")
    return logger


def set_argparsing():
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--source', type=str, nargs='+',required=True)
    parser.add_argument('--rename', type=str, nargs='+',default="")
    args = parser.parse_args()
    return args

class NER_Predictor:
    def __init__(self) :
        self.ner = Taskflow("ner", entity_only=True, batch_size=BATCH_SIZE)

    def __call__(self, batch):
        results = self.ner([text[:800] for text in batch['output']])
        # return {'results': [str(Counter(result)) for result in results], 
        #     'output': batch['output']}
        
        return {'results': [str(Counter(result)) for result in results], 
            'output': batch['output'], 
            'uid': batch['uid']}
     
     
def split_data_by_chunk(data_jsonl_path,output_root,chunk_size=1000000):
    # 读取 JSON 文件并逐块处理
    for i, chunk in enumerate(pd.read_json(data_jsonl_path, lines=True, chunksize=chunk_size)):
        filename = f'{output_root}/part{i}.parquet'
        table = pa.Table.from_pandas(chunk)
        pq.write_table(table, filename)
      
logger = set_logging()
args = set_argparsing()
DATA_NAME_LIST = args.source

ray.init(num_cpus=96, num_gpus=8)
ray.data.DataContext.get_current().execution_options.verbose_progress = True
for DATA_NAME in DATA_NAME_LIST:
    print(f"Start processing {DATA_NAME}...")
    if not os.path.exists(f'{DATA_NAME}/raw/'):
        input_path = f'{DATA_NAME}/trans_data.jsonl'
        output_path= f'{DATA_NAME}/raw/'
        os.mkdir(output_path)
        
        
        split_data_by_chunk(input_path,output_path,chunk_size=500000)
        
        # logger.info("Start reading jsonl data.")
        # df = pd.read_json(input_path, lines=True)
        # logger.info("Finish reading jsonl data.")
        # if args.rename != '':
        #     df.rename(columns={args.rename: 'output'}, inplace=True)
        # print(df.columns) 

        # ddf = da.from_pandas(df, chunksize=500000)
        # ddf.to_parquet(output_path)




    ner = Taskflow("ner", entity_only=True, batch_size=BATCH_SIZE)
    input_path = f'{DATA_NAME}/raw/'
    output_path = f'{DATA_NAME}/tags/'

    '''
    for file in glob.glob(f'{input_path}*'):
        print(file)
    # ds = ray.data.read_parquet('data_100000.parquet')
    '''

    ds = ray.data.read_parquet([file for file in glob.glob(f'{input_path}*')])
    logger.info("Finish data reading.")

    print(ds)
    print(ds.schema())
    print(ds.materialize().stats())

    predictions=ds.map_batches(
        NER_Predictor,
        num_gpus=1,
        batch_size=BATCH_SIZE,
        concurrency=8,
    )
    logger.info("Finish processing.")

    predictions.write_parquet(output_path)
    print(f"Finish writing {DATA_NAME} output files.")