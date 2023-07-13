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
from functools import reduce
import sys
sys.path.append('../')

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

def aggregate(df, df_ref, df_volumes):

    # Filter
    df = df.where((df.type != 'other') & ~df.prev.isin(['other-other', 'other-empty']))

    # Aggregate
    df = df.groupBy('date', 'prev', 'curr', 'date').agg(sum('count').alias('count')).cache()
    initial_links = df.count()

    # Match on ids and volumes
    # Match on page id first
    df = df.join(df_ref.select(col('page').alias('prev'), col('page_id').alias('id_prev')),
                                         on='prev') \
           .join(df_ref.select(col('page').alias('curr'), col('page_id').alias('id_curr')),
                                         on='curr')
    # Match on volumes
    df = df.join(df_volumes.select('date', col('page_id').alias('id_prev'), col('volume').alias('volume_prev')),
                  on=['date', 'id_prev'], how='left') \
           .join(df_volumes.select('date', col('page_id').alias('id_curr'), col('volume').alias('volume_curr')),
                  on=['date', 'id_curr'], how='left')
    df = df.select('date', 'prev', 'curr', coalesce('volume_prev', 'prev').alias('volume_prev'),
                                           coalesce('volume_curr', 'curr').alias('volume_curr'), 'count')
    df = df.where(
        df.volume_prev.isin(['tail', 'core', 'other-search', 'other-internal', 'other-external'])\
        & df.volume_curr.isin(['tail', 'core', 'other-search', 'other-internal', 'other-external'])).cache()
    final_links = df.count()

    print(f"Loss = {100 - initial_links / final_links * 100} %")

    return df

def make_links_dataset(ys, ms, spark_session, path, ref_path, save_path):

    # Make dates
    months = [str(m) if m / 10 >= 1 else f"0{m}" for m in ms]
    dates = [f"{year}-{month}" for year in ys for month in months]

    # Make ref datasets
    df_ref = spark.read.parquet(ref_path).withColumn('project', lit('en'))
    df_high_volume = extract_volume(df_ref.where(df_ref.date.isin([dates])), high=True).select(
        'date', 'page_id', 'page', lit('core').alias('volume'))
    df_low_volume = extract_volume(df_ref.where(df_ref.date.isin([dates])), high=False).select(
        'date', 'page_id', 'page', lit('tail').alias('volume'))
    df_volumes = df_high_volume.union(df_low_volume)

    pd_compls = pd.DataFrame({'date':[i for j in [[d]*3 for d in dates] for i in j],
                              'page_id': ['-1', '-2', '-3']*len(dates),
                              'page': ['other-search', 'other-internal', 'other-external']*len(dates)})
    pd_compls['volume'] = pd_compls['page']
    df_compl = spark.createDataFrame(pd_compls)
    df_volumes = df_volumes.union(df_compl)

    df_ref = df_ref.select('page', 'page_id').distinct()
    df_ref = df_ref.union(df_compl.select('page', 'page_id'))

    # Download data
    dfs = setup_data(ys, ms, spark_session, path)
    df_clickstream = aggregate(dfs, df_ref, df_volumes).cache()
    df_clickstream.write.parquet(os.path.join(save_path, 'clickstream_volume.parquet'))
    df_clickstream_fluxes = df_clickstream.groupBy('date', 'volume_prev', 'volume_curr').agg(sum('count').alias('agg_counts'),
                                                                                     count('*').alias(
                                                                                         'nb_links')).cache()
    df_clickstream_fluxes.write.parquet(os.path.join(save_path, 'clickstream_fluxes.parquet'))





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

