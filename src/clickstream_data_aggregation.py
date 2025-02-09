import argparse
import os

os.environ["JAVA_HOME"] = "/lib/jvm/java-11-openjdk-amd64"

import time
import os
import pickle
import pyspark
from pyspark.sql import *
from pyspark.sql.functions import *

from tqdm import tqdm
from operator import add
from functools import reduce
import sys
sys.path.append('../')
from src.pages_groups_extraction import extract_volume

import pandas as pd
import numpy as np

def setup_data(years, months, spark_session, path="/scratch/descourt/clickstream/en"):
    """
    Load and prepare wikipedia projects clickstream data for given year and month
    :return pyspark dataframe
    """

    def read_file(f_n, date):
        print(f"loading {f_n}")
        df = spark_session.read.csv(f_n, sep='\t')
        return df.selectExpr("_c0 as prev", "_c1 as curr", "_c2 as type", "_c3 as count")\
            .withColumn('date', lit(date))
    project = path.split('/')[-1]
    files_names = [os.path.join(path, f"clickstream-{project}wiki-{year}-{month}.tsv.gz") for year in years for month in months]
    dates = [f"{year}-{month}" for year in years for month in months]
    start = time.time()
    dfs = [read_file(f, d) for f, d in zip(files_names, dates)]
    df = reduce(DataFrame.unionAll, dfs)
    print(f"Elapsed time {time.time() - start} s")
    return df

def aggregate(df, df_volumes):

    # Filter
    df = df.where((df.type != 'other') & ~df.prev.isin(['other-other', 'other-empty']))

    # Aggregate
    df = df.groupBy('date', 'prev', 'curr').agg(sum('count').alias('count')).cache()
    initial_links = df.count()
    print(initial_links)

    # Match on volumes
    df = df.join(df_volumes.select('date', col('page').alias('prev'), col('volume').alias('volume_prev')),
                  on=['date', 'prev']) \
           .join(df_volumes.select('date', col('page').alias('curr'), col('volume').alias('volume_curr')),
                  on=['date', 'curr'])
    final_links = df.count()
    print(final_links)

    print(f"Loss = {100 - final_links / initial_links * 100} %")

    return df

def make_links_dataset(ys, ms, spark_session, path, ref_path, save_path):

    # Make dates
    months = [str(m) if m / 10 >= 1 else f"0{m}" for m in ms]
    dates = [f"{year}-{month}" for year in ys for month in months]

    # Make ref datasets
    df_ref = spark.read.parquet(ref_path).withColumn('project', lit('en'))
    df_high_volume = extract_volume(df_ref.where(df_ref.date.isin(dates)), high=True).select(
        'date', 'page_id', 'page', lit('core').alias('volume'))
    df_low_volume = extract_volume(df_ref.where(df_ref.date.isin(dates)), high=False).select(
        'date', 'page_id', 'page', lit('tail').alias('volume'))
    df_volumes = df_high_volume.union(df_low_volume)

    pd_compls = pd.DataFrame({'date':[i for j in [[d]*3 for d in dates] for i in j],
                              'page_id': ['-1', '-2', '-3']*len(dates),
                              'page': ['other-search', 'other-internal', 'other-external']*len(dates)})
    pd_compls['volume'] = pd_compls['page']
    df_compl = spark.createDataFrame(pd_compls)
    df_volumes = df_volumes.union(df_compl)


    # Download data
    dfs = setup_data(ys, months, spark_session, path)
    df_clickstream = aggregate(dfs, df_volumes).cache()
    df_clickstream.write.parquet(os.path.join(save_path, 'clickstream_volumes.parquet'))
    df_clickstream = df_clickstream.groupBy('date', 'curr', 'volume_curr') \
                                    .pivot('volume_prev').sum('count').fillna(0) \
                                    .withColumn('total-external', reduce(add, [col('other-external'), col('other-search')])) \
                                    .withColumn('total', reduce(add, [col('other-external'), col('other-search'), col('core'), col('tail')])) \
                                    .drop('other-internal')
    df_clickstream.write.parquet(os.path.join(save_path, 'clickstream_grouped.parquet'))





if __name__ == '__main__':
    conf = pyspark.SparkConf().setMaster("local[5]").setAll([
        ('spark.driver.memory', '100G'),
        ('spark.executor.memory', '100G'),
        ('spark.driver.maxResultSize', '0'),
        ('spark.executor.cores', '5'),
        ('spark.local.dir', '/scratch/descourt/spark')
    ])
    # create the session
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    # create the context
    sc = spark.sparkContext
    sc.setLogLevel('ERROR')

    parser = argparse.ArgumentParser(
        description='Wikipedia clickstream processing'
    )

    parser.add_argument('--y',
                        help='year',
                        nargs='+',
                        type=int,
                        default=[2022])
    parser.add_argument('--m',
                        help='months',
                        nargs='+',
                        type=int,
                        default=[1, 2, 3])
    parser.add_argument('--p',
                        help='path',
                        type=str,
                        default="/scratch/descourt/clickstream/en")
    parser.add_argument('--rp',
                        help='reference path',
                        type=str,
                        default="/scratch/descourt/processed_data/en/pageviews_en_2015-2023.parquet")
    parser.add_argument('--sp',
                        help='save path',
                        type=str,
                        default="/scratch/descourt/processed_data/clickstream/en")

    args = parser.parse_args()

    os.makedirs(args.sp, exist_ok=True)

    make_links_dataset(ys=args.y, ms=args.m, spark_session=spark,
                       path=args.p, ref_path=args.rp, save_path=args.sp)

