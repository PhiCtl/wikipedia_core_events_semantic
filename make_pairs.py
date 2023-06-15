import os
os.environ["JAVA_HOME"] = "/lib/jvm/java-11-openjdk-amd64"
# Because otherwise custom modules import errors
import sys
from tqdm import tqdm
sys.path.append('wikipedia_core_events_semantic/')

import pandas as pd
import numpy as np

from pyspark.sql.functions import *
from pyspark import SparkContext
from pyspark.sql.types import ArrayType, IntegerType
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

    # Extract wanted editions
    assert( 0 <= args.precision <= 1.0), 'Precision must be between 0 and 1'
    match_langs = extract_pairs(precision=args.precision)
    editions = match_langs.select(col('lang1').alias('project'))\
                            .unionAll(match_langs.select(col('lang2').alias('project')))\
                            .dropDuplicates(['project'])
    projects = [p['project'] for p in editions.select('project').collect()]

    # Download needed data
    dfs = setup_data(years=args.years, months=args.months, spark_session=spark,
                     path="/scratch/descourt/raw_data/pageviews")

    df_filt = filter_data(dfs, projects, dates=dates)
    df_agg = aggregate_data(df_filt)
    df_agg.write.parquet(os.path.join(args.save_path, f"pageviews_agg_all.parquet"))
    match_langs.write.parquet(os.path.join(args.save_path, f"pairs_{args.precision}.parquet"))





