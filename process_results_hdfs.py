import os
import json
import ray
import glob
import pandas as pd
from collections import Counter

jsonl_path = "/mnt/data/user/tc_agi/lfy/ner_baike_chinese_new_all.jsonl"
parquet_path = "/mnt/data/user/tc_agi/zj/baike_chinese_new_all_parquet"

'''
cnt = 0
file_num = 0
with open(jsonl_path, 'r') as fin:
    dicts = []
    for line in fin:
        obj = json.loads(line)
        dicts.append(obj)

        cnt += 1
        if cnt % 800000 == 0:
            print(cnt)
            df = pd.DataFrame(data=dicts)
            df.to_parquet(f'{parquet_path}/{file_num}.parquet')
            dicts = []
            file_num += 1
    df = pd.DataFrame(data=dicts)
    df.to_parquet(f'{parquet_path}/{file_num}.parquet')
    dicts = []
    file_num += 1
print(cnt)
'''

class MapData:
    def __init__(self) :
        pass

    def __call__(self, batch):
        from collections import Counter
        for results, text in zip(batch['entity'], batch['text']):
            results = eval(results)
            tags = results.most_common(3)

            for tag in tags:
                if tag[1] < 2: break

                file_dir = f'output_multi/{tag[0][1]}/{tag[0][0]}'
                if not os.path.exists(file_dir):
                    os.makedirs(file_dir)

                with open(f'{file_dir}/baike_chinese_new_all.jsonl', 'a') as fout:
                    fout.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
        return {"result": batch['text']}    

ds = ray.data.read_parquet([file for file in glob.glob(f'{parquet_path}/*')])
# ds = ray.data.read_parquet(f'{parquet_path}/0.parquet')

print(ds)
print(ds.schema())
print(ds.materialize().stats())

predictions=ds.map_batches(
    MapData,
    batch_size=1000,
    concurrency=64,
)

# predictions.write_json(output_path, force_ascii=False)
predictions.write_parquet('nouse.parquet')
