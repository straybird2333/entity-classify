import ray
import argparse
# import dask.dataframe as da
import pandas as pd
from paddlenlp import Taskflow
import glob
import logging
from collections import Counter
import os

BATCH_SIZE = 128 # 666 error? 700 error
DATA_NAME = None


def set_logging(): # 设置logger
    logging.basicConfig(level = logging.DEBUG,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s', filename='logging.txt')
    logger = logging.getLogger(__name__)
    # logger.info("Start print log") logger.debug("Do something") logger.warning("Something maybe fail.") logger.info("Finish")
    return logger


def set_argparsing():
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--source', type=str, nargs='+',required=True)
    parser.add_argument('--rename', type=str, default="")
    parser.add_argument('--stage', type=str, default="tags") # 分类只需要output字段，从 raw,tags,final里选择都可以
    args = parser.parse_args()
    return args

class CLS_Predictor:
    def __init__(self) :
        schema = ['农业农学', '娱乐', '教育', '法律', '旅游出行', '金融地产', '家居生活', '能源矿产', '自然科学', '时尚美容', '医学健康', '人文社科', '文化艺术', '科技与互联网', '体育运动', '工业制造', '军事', '社会活动','交通运输','公共管理','情感心理']

        self.cls = Taskflow("zero_shot_text_classification", schema=schema,batch_size=BATCH_SIZE)

    def __call__(self, batch):
        input_text=[text[:256] for text in batch['output']]
        results = self.cls(input_text)
        label_list=[]
        for item in results:
            try:
                label=item['predictions'][0]['label']
            except: ## 无法生成的
                label='其他'
            label_list.append(label)
            
            
        output_dict = {key: value for key, value in batch.items()}
    
        output_dict['domain'] = label_list
        return output_dict


logger = set_logging()
args = set_argparsing()
DATA_NAME_LIST = args.source
STAGE = args.stage
ray.init(num_cpus=96, num_gpus=8)
ray.data.DataContext.get_current().execution_options.verbose_progress = True
logger.info("Finish ray preparation.")
for DATA_NAME in DATA_NAME_LIST:
    print(f"starting processing {DATA_NAME}...")
    input_path = f'{DATA_NAME}/{STAGE}/'
    output_path = f'{DATA_NAME}/tag-domain/'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
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
        CLS_Predictor,
        num_gpus=1,
        batch_size=BATCH_SIZE,
        concurrency=8,
    )
    

    predictions.write_parquet(output_path)
    print("Finish writing output files.")
    
