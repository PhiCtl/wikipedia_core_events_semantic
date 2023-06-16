import os
from itertools import combinations

from pages_groups_extraction import extract_volume

os.environ["JAVA_HOME"] = "/lib/jvm/java-11-openjdk-amd64"
# Because otherwise custom modules import errors
import sys
from tqdm import tqdm
sys.path.append('wikipedia_core_events_semantic/')

import pandas as pd
import numpy as np

from pyspark.sql.functions import *
from pyspark import SparkContext
from pyspark.sql.types import ArrayType, IntegerType, StringType
import pyspark

from data_aggregation import*

def extract_pairs(path='/scratch/descourt/metadata/akhils_data/wiki_nodes_bsdk_phili_2022-11.parquet',
                  precision=0.95):

    def compute_ratio(item):
        return Row(
            lang1=item[0][0].split('wiki')[0],
            lang2=item[1][0].split('wiki')[0],
            ratio=float(np.abs(np.log(item[0][1] / item[1][1]))))

    df_metadata_all = spark.read.parquet(path)
    nb_articles_per_languages = df_metadata_all.groupBy('wiki_db').agg(countDistinct('page_id').alias('nb_articles'))
    nb_articles_per_languages = nb_articles_per_languages.where(nb_articles_per_languages.wiki_db != 'nostalgiawiki')

    # make pairs
    pairs = nb_articles_per_languages.rdd.map(tuple).cartesian(nb_articles_per_languages.rdd.map(tuple))
    ratio_df = pairs.map(lambda r: compute_ratio(r)).toDF()
    matching_lang = ratio_df.where((col('ratio') != 0) & (col('ratio') <= -log(lit(precision))))\
                            .select( array_sort(array(col('lang1'), col('lang2'))).alias('pairs'), 'lang1', 'lang2')\
                            .dropDuplicates(['pairs']).cache()
    return matching_lang

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--precision',
                        type=float,
                        default=0.95)
    parser.add_argument('--memory',
                        default=70,
                        type=int,
                        choices=[30, 50, 70, 100, 120])
    parser.add_argument('--cores',
                        type=int,
                        choices=[1,2,3,4,5,6,7,8,9,10],
                        default=5)

    parser.add_argument('--years',
                        nargs='+',
                        type=str,
                        default=['2022'])
    parser.add_argument('--months',
                        nargs='+',
                        type=int,
                        default=[str(m) if (m)/10 >= 1 else f"0{m}" for m in range(1,13)])

    parser.add_argument('--spark_dir',
                        type=str,
                        default='/scratch/descourt/spark')
    parser.add_argument('--save_path',
                        type=str,
                        default='/scratch/descourt/processed_data/multieditions')
    parser.add_argument('--metadata_dir',
                        type=str,
                        default='/scratch/descourt/metadata/akhils_data/wiki_nodes_bsdk_phili_2022-11.parquet')

    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    conf = pyspark.SparkConf().setMaster(f"local[{args.cores}]").setAll([
        ('spark.driver.memory', f'{args.memory}G'),
        ('spark.executor.memory', f'{args.memory}G'),
        ('spark.driver.maxResultSize', '0'),
        ('spark.executor.cores', f'{args.cores}'),
        ('spark.local.dir', args.spark_dir)
    ])
    # create the session
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    # create the context
    sc = spark.sparkContext
    sc.setLogLevel('ERROR')

    # Dates
    dates = [f"{year}-{month}" for year in args.years for month in args.months]

    # Meta Data
    df_metadata = spark.read.parquet(args.metadata_dir)

    ## 1 - Extract editions pairs with matching number of articles
    assert( 0 <= args.precision <= 1.0), 'Precision must be between 0 and 1'
    match_langs = extract_pairs(precision=args.precision)
    editions = match_langs.select(col('lang1').alias('project'))\
                            .unionAll(match_langs.select(col('lang2').alias('project')))\
                            .dropDuplicates(['project'])
    dfs_pairs = match_langs.union(match_langs.select(array(col('pairs')[1], col('pairs')[0]).alias('pairs'),
                                                     'lang1', 'lang2'))
    projects = [p['project'] for p in editions.select('project').collect()]

    ## 2 - Download needed data and filter per project
    dfs = setup_data(years=args.years, months=args.months, spark_session=spark,
                     path="/scratch/descourt/raw_data/pageviews")

    df_filt = filter_data(dfs, projects, dates=dates)
    df_agg = aggregate_data(df_filt).cache()

    # Extract volumes
    df_high_volume = extract_volume(df_agg, high=True).select('date', 'page_id', 'item_id', 'project')
    df_low_volume = extract_volume(df_agg, high=False).select('date', 'page_id', 'item_id', 'project')

    ## 3 - Match pairs
    def make_pairs(langs):
        return [list(p) for p in combinations(langs, 2)]


    make_pairs_udf = udf(make_pairs, ArrayType(ArrayType(StringType())))
    df_high_pairs = df_high_volume.groupby('date', 'item_id')\
                                  .agg(collect_set('project').alias('langs'))\
                                  .select('date','item_id',explode(make_pairs_udf('langs')).alias('pairs'))

    # Retrieve since when article is in core or in tail
    w = Window.partitionBy('project', 'item_id').orderBy(asc('date'))
    df_high_volume = df_high_volume.withColumn('when_in_core', array_sort(collect_set('date').over(w)))
    df_whenintail = df_low_volume.select(add_months(col('date'), 1).alias('date'), 'item_id', 'project',
                                         array_sort(collect_set('date').over(w)).alias('when_in_tail'))

    # Find pairs in dataset which belong to ## 1 - and are both for the first time in the core at the given date d
    df_high_pairs = df_high_pairs.join(dfs_pairs, on='pairs') \
                                .select('date', 'item_id', 'pairs', explode('pairs').alias('project')) \
                                .join(df_high_volume, on=['project', 'item_id', 'date'])\
                                .where(col('date') == col('when_in_core')[0])
    df_high_pairs_agg = df_high_pairs.groupBy('date', 'item_id', 'pairs').\
                                     agg(count('*').alias('first_time_core'))
    df_high_pairs = df_high_pairs.join(df_high_pairs_agg, on=['date', 'item_id', 'pairs'])\
                                 .where('first_time_core = "2"').drop('first_time_core')

    ## 4 - Find matching pairs for which exactly 1 article out of two was in the tail before
    w = Window.partitionBy('pairs', 'item_id', 'date')
    df_high_pairs = df_high_pairs.join(df_whenintail, on=['project', 'item_id', 'date'], how='left') \
        .select(coalesce(col('when_in_tail'), array(to_date(lit('2023-01'), 'yyyy-MM'))).alias('when_in_tail'),
                'project', 'item_id', 'date', 'pairs', 'when_in_core')
    df_high_pairs = df_high_pairs\
        .select('date', 'item_id', 'pairs', 'project',
                when((element_at(col('when_in_tail'), -1) == add_months(col('date'), -1)), 1).otherwise(0).alias('was_in_tail')) \
        .select(sum('was_in_tail').over(w).alias('intail'), 'date', 'project', 'item_id', 'was_in_tail')

    df_agg.write.parquet(os.path.join(args.save_path, f"pageviews_agg_all.parquet"))
    match_langs.write.parquet(os.path.join(args.save_path, f"pairs_{args.precision}.parquet"))
    df_high_pairs.where('intail = "1"').write.parquet(os.path.join(args.save_path, f"matchedpairs_{args.precision}.parquet"))





