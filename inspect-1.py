import os
import json
import pandas as pd
import pickle

# baike_chinese_new_all text
# gov_safety text
# people_daily_new output
# xuexiqiangguo output
path_1='/data/ner_classify/stack_exchange_qa/final/3_000000_000000.parquet'
df=pd.read_parquet(path_1,engine='pyarrow')
print(df.columns)
pass