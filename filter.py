import os, glob, json
import logging
from tqdm import tqdm
import pandas as pd
import argparse


def set_logging():
    logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    # logger.info("Start print log")
    # logger.debug("Do something")
    # logger.warning("Something maybe fail.")
    # logger.info("Finish")
    return logger


def filter(df, base_word, words, input_path):
    logger.info("Start reading parquet")
    # df = pd.read_parquet([file for file in glob.glob('baike_new_all_tags/*.parquet')[:3]], engine='pyarrow')
    df = pd.read_parquet(f'{input_path}/final/', engine='pyarrow')
    logger.info("Finish reading parquet %s" % str(df.shape))

    fdf = df.loc[(df['ent0'].isin(words)) | (df['ent1'].isin(words)) | (df['ent2'].isin(words))]
    fdf = fdf[['text', 'ent0', 'typ0', 'ent1', 'typ1', 'ent2', 'typ2']]
    print(fdf.shape)
    fdf.to_csv(f'{base_word}/{base_word}.csv')


def get_ent_list(input_path):
    # build the entity count list
    df = pd.read_parquet(f'{input_path}/final/', engine='pyarrow')
    # df = pd.read_parquet(f'{input_path}/final-tag-domain/', engine='pyarrow')

    cdf = pd.concat([df['ent0'], df['ent1'], df['ent2']])
    # filter(df, word, output_path)
    cdf = cdf.value_counts()
    cdf.to_csv(f'{input_path}/count.csv')


def set_argparsing():
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--source', type=str, required=True)
    parser.add_argument('--word', type=str, required=True)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    logger = set_logging()
    pd.set_option('display.max_columns', None)
    args = set_argparsing()

    input_path = args.source
    base_word = args.word

    if not os.path.exists(f'{input_path}/count.csv'):
        get_ent_list(input_path)

    if not os.path.exists(base_word+"/"): os.mkdir(base_word)
    word_lists = []

    if not os.path.exists(f"{base_word}/list.txt"):
        df = pd.read_csv(f'{input_path}/count.csv')
        df.columns = ['name', 'count']
        words = list(df[df['name'].fillna('').str.contains(base_word)]['name'])

        logger.info("All words: " + str(words))
        # for word in words:
            # ans = input(f'{word}? Y(y)/N(n) ')
            # if ans in ['y', 'Y']:
            #     word_lists.append(word)
            # word_lists.append(word)

        logger.info("Selected words: " + str(word_lists))
        with open(f"{base_word}/list.txt", "w") as fout:
           fout.write(json.dumps({"all": words, "select": word_lists}, ensure_ascii=False)) 
    else:
        with open(f"{base_word}/list.txt", "r") as fin:
            obj = json.loads(fin.read())
            word_lists = obj["select"]

    word_lists = [base_word]
    logger.info("Final words: " + str(word_lists))

    filter(pd, base_word, word_lists, input_path)