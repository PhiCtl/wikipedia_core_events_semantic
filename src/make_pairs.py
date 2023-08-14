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

def extract_pairs(path_nbarticles='/scratch/descourt/metadata/akhils_data/wiki_nodes_bsdk_phili_2022-11.parquet',
                  path_nbviews='/scratch/descourt/raw_data/pageviews',
                  precision=0.90):

    def compute_ratio(item):
        return Row(
            lang1=item[0][0],
            lang2=item[1][0],
            ratio_articles=float(np.abs(np.log(item[0][1] / item[1][1]))),
            ratio_views=float(np.abs(np.log(item[0][2] / item[1][2]))),
            nbarticles_1=item[0][1],
            nbarticles_2=item[1][1],
            nbviews_1=item[0][2],
            nbviews_2=item[1][2],
        )

    df_projects = spark.read.parquet(path_nbarticles)\
                       .select(split('wiki_db', 'wiki')[0].alias('project'), 'page_id', 'page_title')
    df_nov22 = setup_data([2022], [11], spark, path=path_nbviews)
    df_nov22 = df_nov22.where(df_nov22.project.contains('.wikipedia')) \
                       .groupBy('project', 'page', 'page_id').agg(sum('counts').alias('counts'))\
                       .select(split('project', '.wikipedia')[0].alias('project'),
                               col('page').alias('page_title'),
                               'page_id',
                               'counts')
    df_filt_nov22 = df_nov22.join(df_projects, on=['project', 'page_title', 'page_id'], how='right')
    df_agg_nov_22 = df_filt_nov22.groupBy('project').agg(count('*').alias('nb_articles'),
                                                         sum('counts').alias('nb_views')) \
                                  .dropna(subset='nb_views').cache()

    pairs = df_agg_nov_22.rdd.map(tuple).cartesian(df_agg_nov_22.rdd.map(tuple))
    ratio_df = pairs.map(lambda r: compute_ratio(r)).toDF()
    matching_lang = ratio_df.where(
        (col('ratio_articles') != 0) & (col('ratio_articles') <= -log(lit(precision)))
        & (col('ratio_views') != 0) & (col('ratio_views') <= -log(lit(precision)))) \
        .select(array_sort(array(col('lang1'), col('lang2'))).alias('pairs'), 'lang1', 'lang2', 'nbarticles_1',
                'nbarticles_2', 'nbviews_1', 'nbviews_2') \
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
                        default='/scratch/descourt/processed_data/pairs')
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

    ## 1 - Extract editions pairs with matching number of articles
    assert( 0 <= args.precision <= 1.0), 'Precision must be between 0 and 1'
    match_langs = extract_pairs(precision=args.precision)
    matching_lang.write.parquet(os.path.join(args.save_path, 'pairs_10perc_nov22.parquet'))

    # selected_langs = matching_lang.sort(desc('nbarticles_1')).limit(7)
    projects = [p['lang'] for p in matching_lang.select(explode('pairs').alias('lang')).distinct().collect()]
    dfs_pairs = matching_lang.select('pairs', 'lang1', 'lang2')\
                              .union(matching_lang.select(array(col('pairs')[1], col('pairs')[0]).alias('pairs'),
                                     'lang1', 'lang2'))

    ## 2 - Download needed data and filter per project and add item id
    dfs = setup_data(years=args.years, months=args.months, spark_session=spark,
                     path="/scratch/descourt/raw_data/pageviews")
    # Meta Data with project, page id and item id (same across editions)
    df_metadata = spark.read.parquet(args.metadata_dir)\
                       .select((split('wiki_db', 'wiki')[0]).alias('project'),
                               'page_id', 'item_id') \
                       .join(matching_lang.select(explode('pairs').alias('project')).distinct(), on='project').cache()


    df_filt = dfs.select(split('project', '.wikipedia')[0].alias('project'),
                         'date', 'page', 'page_id', 'counts') \
                 .groupBy('date', 'project', 'page', 'page_id')\
                 .agg(sum('counts').alias('tot_count_views'))
    df_agg = df_filt.join(df_metadata, on=['project', 'page_id']) \
                    .groupBy('date', 'project', 'page_id', 'item_id')\
                    .agg(sum('counts').alias('tot_count_views'))

    # Make ranking for volume computation
    window = Window.partitionBy('project', 'date').orderBy(col("tot_count_views").desc())
    df_agg = df_agg.withColumn("rank", row_number().over(window))

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
        .select(coalesce(col('when_in_tail'), array(to_date(lit('2024-01'), 'yyyy-MM'))).alias('when_in_tail'),
                'project', 'item_id', 'date', 'pairs', 'when_in_core')
    df_high_pairs = df_high_pairs\
        .select('date', 'item_id', 'pairs', 'project',
                when((element_at(col('when_in_tail'), -1) == add_months(col('date'), -1)), 1).otherwise(0).alias('was_in_tail')) \
        .select(sum('was_in_tail').over(w).alias('intail'), 'date', 'project', 'item_id', 'was_in_tail')

    df_high_pairs.where('intail = "1"').write.parquet(os.path.join(args.save_path, f"matchedpairs_{args.precision}.parquet"))





