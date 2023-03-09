import os
os.environ["JAVA_HOME"] = "/lib/jvm/java-11-openjdk-amd64"

import time
import os
from functools import reduce
import argparse
import pyspark
from pyspark.sql import *
from pyspark.sql.functions import *

conf = pyspark.SparkConf().setMaster("local[*]").setAll([
                                   ('spark.driver.memory','24g'),
                                   ('spark.driver.maxResultSize', '8G')
                                  ])
# create the session
spark = SparkSession.builder.config(conf=conf).getOrCreate()
# create the context
sc = spark.sparkContext


def setup_data(years, months, path="/scratch/descourt/pageviews", project='en'):
    """
    Load and prepare wikipedia projects pageviews data for given year and month
    :return pyspark dataframe
    """

    def read_file(f_n, date):
        print(f"Loading {f_n}")
        df = spark.read.csv(f_n, sep=r' ')
        return df.selectExpr("_c0 as project", "_c1 as page", "_c2 as null", "_c3 as access_type", "_c4 as count",
                             "_c5 as idontknow").withColumn('date', lit(date)).select('project', 'page', 'count',
                                                                                      'date')

    files_names = [os.path.join(path, f"pageviews-{year}{month}-user.bz2") for year in years for month in months]
    dates = [f"{year}-{month}" for year in years for month in months]
    start = time.time()
    dfs = [read_file(f, d) for f, d in zip(files_names, dates)]
    df = reduce(DataFrame.unionAll, dfs)
    print(f"Elapsed time {time.time() - start} s")
    return df


def filter_data(df, project, specials, dates):
    """
    Filter in wanted data from initial dataframe
    """

    df_filt = df.where(f"project = '{project}'") \
                .filter(df.date.isin(dates)) \
                .select(lower(col('page')).alias('page'), 'project', 'count', 'date')
    df_filt = df_filt.filter(~df_filt.page.isin(specials))

    return df_filt


def aggregate_data(df):
    """
    TODO
    """
    # rank pages for each date
    df_agg = df.groupBy("date", "page") \
                .agg(sum("count").alias("tot_count_views")) \
                .sort(['date', "tot_count_views"], ascending=False)

    return df_agg


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--y',
                        help="Years to process",
                        type=str,
                        nargs='+',
                        default=['2021'])
    parser.add_argument('--m',
                        help="Months to process for each year",
                        type=int,
                        nargs='+',
                        default=[1, 2, 3])
    parser.add_argument('--path',
                        type=str,
                        default="/scratch/descourt/pageviews")
    parser.add_argument('--save_path',
                        type=str,
                        default="/scratch/descourt/processed_data")

    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)

    months = [str(m) if m / 10 >= 1 else f"0{m}" for m in args.m]
    dates = [f"{year}-{month}" for year in args.y for month in months]

    dfs = setup_data(years=args.y, months=months)
    df_filt = filter_data(dfs, 'en.wikipedia', ['main_page', 'special:search', '-'], dates=dates)
    df_filt = df_filt.cache()

    df_agg = aggregate_data(df_filt)
    df_agg.write.parquet(os.path.join(args.save_path, "pageviews_agg.parquet"))

    print("Done")









