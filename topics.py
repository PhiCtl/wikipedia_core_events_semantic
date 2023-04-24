import pandas as pd
from ast import literal_eval
import gzip
import os
import json
from tqdm import tqdm
import pickle

import numpy as np

"""
Temporary file with code to gzip json files
"""

def zip_json(dir_, dir_out):
    ls_dirs = os.listdir(dir_)

    for subdir in tqdm(ls_dirs):
        for file in os.listdir(os.path.join(dir_, subdir)):
            path_in = os.path.join(dir_, subdir, file)
            with open(path_in, 'r') as f:
                data = f.read().replace('\n', ',')
                data = literal_eval(data)
            os.makedirs(os.path.join(dir_out, subdir), exist_ok=True)
            path_out = os.path.join(dir_out, subdir, file)
            with gzip.open(path_out + ".json.gz", 'wt', encoding='utf-8') as f:
                json.dump(data, f)

def parse_topics(path_in="/home/descourt/topic_embeddings/topics_enwiki.tsv.zip",
                 path_out='/home/descourt/topic_embeddings/topics-enwiki-20230320-parsed.parquet'):

    df_topics = pd.read_csv(path_in,
                            sep='\t')  # , converters={'topics': literal_eval} )
    df_topics['topics'] = df_topics['topics'].apply(
        lambda st: [literal_eval(s.strip())['topic'] for s in st[1:-1].split("\n")])
    df_topics['topics'] = df_topics['topics'].apply(
        lambda l: [x.lower().strip().replace("'", '') for x in l])
    df_topics['page_title'] = df_topics['page_title'].apply(lambda p: p.lower() if isinstance(p, str) else str(p))
    df_topics['topics_unique'] = df_topics['topics'].apply(lambda l: l[0])
    df_topics['topics_specific_unique'] = df_topics['topics'].apply(
        lambda l: l[0] if len([t for t in l if not '*' in t]) == 0 else [t for t in l if not '*' in t][0])
    df_topics['weight'] = df_topics['topics'].apply(lambda l: 1 / len(l))
    df_topics = df_topics.explode('topics')
    df_topics.to_parquet(path_out, engine='fastparquet')

def parse_embeddings(path_in="/home/descourt/topic_embeddings/article-description-embeddings_enwiki-20210401-fasttext.pickle",
                     path_out='/home/descourt/topic_embeddings/embeddings-20210401.parquet'):
    with open(path_in, 'rb') as handle:
        dict_embeddings = pickle.load(handle)

    page_ids = [i for i in dict_embeddings.keys()]
    embeds = np.array([dict_embeddings[i].tolist() for i in page_ids])
    df_embeds = pd.DataFrame(embeds)
    df_embeds['page_id'] = page_ids
    df_embeds.rename({i: str(i) for i in df_embeds.columns}, inplace=True, axis=1)
    df_embeds.to_parquet(path_out, engine='fastparquet')

if __name__ == '__main__':

    dir_ = "/scratch/descourt/wiki_dumps/extracted"
    dir_out = "/scratch/descourt/wiki_dumps/extracted_zip"
    zip_json(dir_, dir_out)