import os
os.environ["JAVA_HOME"] = "/lib/jvm/java-11-openjdk-amd64"

import time
import os
from functools import reduce
import argparse
import pyspark
from pyspark.sql import *
from pyspark.sql.functions import *

conf = pyspark.SparkConf().setMaster("local[10]").setAll([
                                   ('spark.driver.memory','70G'),
                                   ('spark.executor.memory', '70G'),
                                   ('spark.driver.maxResultSize', '0'),
                                    ('spark.executor.cores', '10')
                                  ])
# create the session
spark = SparkSession.builder.config(conf=conf).getOrCreate()
# create the context
sc = spark.sparkContext
sc.setLogLevel('ERROR')


def setup_data(years, months, path="/scratch/descourt/pageviews"):
    """
    Load and prepare wikipedia projects pageviews data for given year and month
    :return pyspark dataframe
    """

    def read_file(f_n, date):
        print(f"loading {f_n}")
        df = spark.read.csv(f_n, sep=r' ')
        return df.selectExpr("_c0 as project", "_c1 as page", "_c2 as page_id", "_c3 as access_type", "_c4 as counts",
                             "_c5 as anonym_user").withColumn('date', lit(date))

    files_names = [os.path.join(path, f"pageviews-{year}{month}-user.bz2") for year in years for month in months]
    dates = [f"{year}-{month}" for year in years for month in months]
    start = time.time()
    dfs = [read_file(f, d) for f, d in zip(files_names, dates)]
    df = reduce(DataFrame.unionAll, dfs)
    print(f"Elapsed time {time.time() - start} s")
    return df

def specials(project):
    if project == 'en.wikipedia':
        return ['main_page', '-']
    # TODO refine for french edition
    elif project == 'fr.wikipedia':
        return ['wikipédia:accueil_principal', '-', 'spécial:recherche' 'special:search']
    elif project == 'es.wikipedia':
        return ['wikipedia:hauptseite', 'spezial:suche', '-', 'special:search', 'wikipedia']
    elif project == 'de.wikipedia':
        return ['wikipedia', 'wikipedia:portada', 'especial:buscar', '-', 'spécial:recherche', 'special:search']

def filter_data(df, project, dates):
    """
    Filter in wanted data from initial dataframe
    """
    specials_to_filt = specials(project)
    df_filt = df.where(f"project = '{project}'") \
                .filter(df.date.isin(dates)) \
                .select(lower(col('page')).alias('page'), 'project', 'counts', 'date', 'page_id')
    df_filt = df_filt.filter(~df_filt.page.contains('user:') & \
             ~df_filt.page.contains('wikipedia:') & \
             ~df_filt.page.contains('file:') & \
             ~df_filt.page.contains('mediawiki:') & \
             ~df_filt.page.contains('template:') & \
             ~df_filt.page.contains('help:') & \
             ~df_filt.page.contains('category:') & \
             ~df_filt.page.contains('portal:') & \
             ~df_filt.page.contains('draft:') & \
             ~df_filt.page.contains('timetext:') & \
             ~df_filt.page.contains('module:') & \
             ~df_filt.page.contains('special:') & \
             ~df_filt.page.contains('media:') & \
             ~df_filt.page.contains('_talk:') & \
             ~df_filt.page.isin(specials_to_filt)\
             & (df_filt.counts >= 1))

    return df_filt


def aggregate_data(df):
    """
    TODO
    """
    # rank pages for each date
    df_agg = df.groupBy("date", "page") \
                .agg(sum("counts").alias("tot_count_views")) \
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
    parser.add_argument('--project',
                        type=str,
                        default='en.wikipedia',
                        help="Project to process")

    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)

    months = [str(m) if m / 10 >= 1 else f"0{m}" for m in args.m]
    dates = [f"{year}-{month}" for year in args.y for month in months]

    dfs = setup_data(years=args.y, months=months)
    df_filt = filter_data(dfs, args.project, dates=dates)

    df_agg = aggregate_data(df_filt)
    df_agg.write.parquet(os.path.join(args.save_path, f"pageviews_agg_{args.project}_{'_'.join(args.y)}.parquet"))

    print("Done")

if __name__ == '__main__':

    main()
    # dfs = spark.read.parquet(
    #     "/scratch/descourt/processed_data/pageviews_agg_en.wikipedia_2016_2017_2018_2019_2020_2021_2022.parquet")
    # df_25 = spark.read.parquet("/scratch/descourt/processed_data/pageviews_agg_en.wikipedia_2015.parquet")
    # dfs.union(df_25).write.parquet(
    #     "/scratch/descourt/processed_data/pageviews_agg_en.wikipedia_2015_2016_2017_2018_2019_2020_2021_2022.parquet")









