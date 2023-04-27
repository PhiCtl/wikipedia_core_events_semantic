import pandas as pd
from ast import literal_eval
import gzip
import os
import json
from tqdm import tqdm
import pickle
import numpy as np
import os

import pyspark
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler

os.environ["JAVA_HOME"] = "/lib/jvm/java-11-openjdk-amd64"

conf = pyspark.SparkConf().setMaster("local[*]").setAll([
                                   ('spark.driver.memory','32'
                                                          'G'),
                                   ('spark.executor.memory', '32G'),
                                   ('spark.driver.maxResultSize', '0'),
                                    ('spark.executor.cores', '10')
                                  ])
# create the session
spark = SparkSession.builder.config(conf=conf).getOrCreate()
# create the context
sc = spark.sparkContext
sc.setLogLevel('ERROR')


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
                     path_out='/home/descourt/topic_embeddings/embeddings-20210401-sp.parquet',
                     path_qid="/home/descourt/topic_embeddings/title_pid-20210901.jsonl.bz2",
                     debug=True):
    # Load embeddings for < 2021-04
    if debug: print("Load embeddings for < 2021-04")
    with open(path_in, 'rb') as handle:
        dict_embeddings = pickle.load(handle)

    # Load qid to page title mapping for < 2021-04
    if debug: print("Load qid to page title mapping for < 2021-04")
    df_qid = spark.read.json(path_qid)
    df_qid = df_qid.withColumn('page_title', lower(regexp_replace('page_title', ' ', '_')))

    # Process embeddings
    if debug: print("Process embeddings pandas")
    page_ids = [i for i in dict_embeddings.keys()]
    embeds = np.array([dict_embeddings[i].tolist() for i in page_ids])
    df_embeds = pd.DataFrame(embeds)
    df_embeds['page_id'] = page_ids
    df_embeds.rename({i: str(i) for i in df_embeds.columns}, inplace=True, axis=1)

    if debug: print("Process embeddings pyspark")
    df_embeds = spark.createDataFrame(df_embeds).join(df_qid, 'page_id')
    assembler = VectorAssembler(inputCols=[c for c in df_embeds.columns if c not in ['page_id', 'page_title']],
                                outputCol='embed')
    df_embeds_vec = assembler.transform(df_embeds).select('page_title', 'embed')
    df_embeds_vec.write.parquet(path_out)


if __name__ == '__main__':
    parse_embeddings()