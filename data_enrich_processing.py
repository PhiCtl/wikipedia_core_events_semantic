import pandas as pd
from ast import literal_eval
import pickle
import numpy as np
import os

import pyspark
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler

from data_aggregation import get_target_id

os.environ["JAVA_HOME"] = "/lib/jvm/java-11-openjdk-amd64"

def find_topic_specific(topics):
    not_starred = [t for t in topics if not '*' in t]
    if len(not_starred) == 0:
        return topics[0]
    else:
        return not_starred[0]
find_topic_specific_udf = udf(find_topic_specific)


def parse_topics(path_in="/scratch/descourt/metadata/topics/topic_en/topics_enwiki.tsv.zip",
                 path_out='/scratch/descourt/metadata/topics/topic_en/topics-enwiki-20230320-parsed.parquet'):

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

def parse_embeddings(path_in="/scratch/descourt/metadata/semantic_embeddings/fr/article-description-embeddings_frwiki-20210401-fasttext.pickle",
                     path_out='/scratch/descourt/metadata/semantic_embeddings/fr/embeddings-fr-20210401.parquet',
                     debug=True):

    # Load embeddings for < 2021-04
    if debug: print("Load embeddings for < 2021-04")
    with open(path_in, 'rb') as handle:
        dict_embeddings = pickle.load(handle)

    # Process embeddings
    if debug: print("Process embeddings pandas")
    page_ids = [i for i in dict_embeddings.keys()]
    embeds = np.array([dict_embeddings[i].tolist() for i in page_ids])
    df_embeds = pd.DataFrame(embeds)
    df_embeds['page_id'] = page_ids
    df_embeds.rename({i: str(i) for i in df_embeds.columns}, inplace=True, axis=1)
    df_embeds.to_parquet(path_out, engine='fastparquet')

    if debug: print("Process embeddings pyspark")
    df_embeds = spark.read.parquet(path_out)
    assembler = VectorAssembler(inputCols=[c for c in df_embeds.columns if c != 'page_id'],
                                outputCol='embed')
    df_embeds_vec = assembler.transform(df_embeds).select('page_id', 'embed')
    df_embeds_vec.write.parquet(path_out.split('.')[0] + "-sp.parquet")

def parse_ORES_scores(path_scores="/scratch/descourt/metadata/quality/ORES_quality_en_March21.json.gz",
                      save_interm=True):

    df_quality = spark.read.json(path_scores)
    rev_ids = [str(i['revision_id']) for i in df_quality.select('revision_id').distinct().collect()]

    mappings = get_target_id(rev_ids, request_type='revisions', request_id='revids')
    if save_interm:
        with open("/scratch/descourt/topics/quality/mappings_rev2page.pickle", "wb") as handle:
            pickle.dump(mappings, handle, protocol=pickle.HIGHEST_PROTOCOL)

    mappings_spark = [(k, v) for k, v in mappings.items()]
    df_matching = spark.createDataFrame(data=mappings_spark, schema=["revision_id", "page_id"])

    df_quality = df_quality.join(df_matching, 'revision_id')
    df_quality.write.parquet(path_scores.split('.')[0] + '.parquet')

def parse_Wikirank_scores(path_in='/scratch/descourt/metadata/quality/wikirank_scores_201807.tsv.zip',
                          project='en'):

    path_out = path_in.split('.')[0] + f'_{project}.parquet'
    for chunk in pd.read_csv(path_in, sep='\t', chunksize=10**6):

        if chunk.loc[chunk['language'] == project].shape[0] >= 1:

            df = chunk.loc[chunk['language'] == project]
            df['revision_id'] = df['revision_id'].astype(str)
            df['language'] = df['language'].astype(str)
            df['page_id'] = df['page_id'].astype(str)
            df['page_name'] = df['page_name'].astype(str)
            df['wikirank_quality'] = df['wikirank_quality'].astype(float)

            if not os.path.isfile(path_out):
                df.to_parquet(path_out, engine='fastparquet', index=False)
            else:
                df.to_parquet(path_out, engine='fastparquet', index=False, append=True)

def parse_metadata(path_in='/scratch/descourt/metadata/akhils_data/wiki_nodes_bsdk_phili_2022-11.parquet',
                   project='en'):

    df_meta = spark.read.parquet(path_in)
    path_out = path_in.split('.')[0] + '_' + project + '.parquet'
    df_meta_filt = df_meta.where(f'wiki_db = "{project}wiki"')
    # Adapt timestamp to our custom timestamp
    if 'creation_date' in df_meta_filt.columns:
        df_meta_filt = df_meta_filt.withColumn('creation_date', concat(split(col('page_creation_timestamp'), '-')[0],
                                                                       lit('-'),
                                                                       split(col('page_creation_timestamp'), '-')[1]))\
                                   .drop('creation_year', 'creation_month')

    if 'topic' in df_meta_filt.columns:
        df_meta_filt = df_meta_filt.where('score >= "0.5"') \
            .select('page_id', 'topic', col('score').cast('float')) \
            .groupBy('page_id') \
            .agg(sort_array(collect_list(struct("score", "topic")), asc=False).alias("topicsList")) \
            .select('page_id', col('topicsList.topic').alias('topics'),
                    find_topic_specific_udf(col('topicsList.topic')).alias('topic')).cache()

    df_meta_filt.write.parquet(path_out)


if __name__ == '__main__':
    conf = pyspark.SparkConf().setMaster("local[*]").setAll([
        ('spark.driver.memory', '70G'),
        ('spark.executor.memory', '70G'),
        ('spark.driver.maxResultSize', '0'),
        ('spark.executor.cores', '5'),
        ('spark.local.dir', '/scratch/descourt/spark')
    ])
    # create the session
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    # create the context
    sc = spark.sparkContext

    parse_metadata(path_in='/scratch/descourt/metadata/akhils_data/wiki_nodes_bios_bsdk_phili_2022-11.parquet')