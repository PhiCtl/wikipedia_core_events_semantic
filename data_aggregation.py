import os

os.environ["JAVA_HOME"] = "/lib/jvm/java-11-openjdk-amd64"

import time
import os
from functools import reduce
import argparse
import pyspark
from pyspark.sql import *
from pyspark.sql.functions import *

from tqdm import tqdm
from functools import reduce

conf = pyspark.SparkConf().setMaster("local[10]").setAll([
    ('spark.driver.memory', '120G'),
    ('spark.executor.memory', '120G'),
    ('spark.driver.maxResultSize', '0'),
    ('spark.executor.cores', '10'),
    ('spark.local.dir', '/scratch/descourt/spark')
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


def filter_data(df, project, dates, remove_false_pos=True):
    """
    Filter in wanted data from initial dataframe
    """
    specials_to_filt = specials(project)
    df_filt = df.where(f"project = '{project}'") \
        .filter(df.date.isin(dates)) \
        .select(lower(col('page')).alias('page'), 'counts', 'date', 'page_id', 'access_type')
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
                             ~df_filt.page.isin(specials_to_filt) \
                             & (df_filt.counts >= 1))

    if remove_false_pos:
        df_filt = collect_false_positive(df_filt)

    return df_filt


def match_ids(df, latest_date, project):
    """
    Match page ids from latest date dataframe with pageids
    Especially for data before 2015-12 because page ids weren't matched
    """

    [y, m] = latest_date.split('-')

    # Select all available pages and their page ids (raw) for project of interest
    df_latest = setup_data([y], [m])  # Download all data

    # Select columns of interest and filter project
    df_latest = df_latest.where((df_latest.project == project) & (df_latest.page_id != 'null')) \
        .select('page', 'page_id', 'counts')

    # Select unique id per page
    w = Window.partitionBy('page').orderBy(col("tot_count_views").desc())
    df_pageids = df_latest.groupBy('page', 'page_id') \
        .agg(sum('counts').alias('tot_count_views')) \
        .withColumn('page_id', first('page_id').over(w))\
        .select('page', 'page_id').distinct().cache()

    # Join on page title to recover page ids if any
    df = df.drop('page_id').where(f"project = '{project}'").join(df_pageids, 'page', 'left')

    return df


def collect_false_positive(df_to_clean, lim=1000, thresh=10000):

    """
    Return data
    """
    # Take top 100 pages (should be a good estimation of actual ranking irr. of redirects)
    df_agg = df_to_clean.groupBy('date', 'page').agg(sum('counts').alias('tot_counts')).sort(desc('tot_counts')).limit(lim).cache()
    # Compute number of counts for mobile versus web
    df_agg = df_agg.join(df_to_clean.where('access_type = "desktop"').groupBy('date', 'page').agg(sum('counts').alias('tot_counts_desktop')), on=['date', 'page'])
    df_agg = df_agg.join(df_to_clean.where('access_type = "mobile-web"').groupBy('date', 'page').agg(sum('counts').alias('tot_counts_mobweb')), on=['date', 'page'])
    # Get false positive titles based on desktop to mobile web views ratio
    df_FP = df_agg.where((col('tot_counts_desktop') / col('tot_counts_mobweb') >= thresh) | (col('tot_counts_desktop') / col('tot_counts_mobweb') <= 1 / thresh)).withColumn('toDelete', lit(1))

    return df_to_clean.join(df_FP.select('date', 'page', 'toDelete'), on=['page', 'date'], how='left').where(col('toDelete').isNull())


def aggregate_data(df, match_ids=True):
    """
    Compute the aggregated number of views for each page per month,
    Filter out views without page id
    Compute page ranking per month according to total aggregated page views
    """
    if match_ids:

        # 1. Aggregate counts by page title and page ids &
        #    Sorting by descending aggregated counts and grouping by page, select first page id
        w = Window.partitionBy('date', 'page').orderBy(col("tot_count_views").desc())
        df_agg = df.groupBy('date', 'page', 'page_id') \
            .agg(sum('counts').alias('tot_count_views')) \
            .withColumn('page_id', first('page_id').over(w))

        # 2. Sorting by descending aggregated counts and grouping by page id, select first page title
        w = Window.partitionBy('date', 'page_id').orderBy(col("tot_count_views").desc())
        df_agg = df_agg.withColumn('page', first('page').over(w))

        # 3. Aggregate counts by page title
        df_agg = df_agg.groupBy('date', 'page').agg(sum('tot_count_views').alias('tot_count_views'),
                                                    first('page_id').alias('page_id'))
        df_agg = df_agg.where(~col('page_id').isNull() & ~(df_agg.page_id == 'null'))
    else:
        df_agg = df.groupBy('date', 'page').agg(sum("counts").alias('tot_count_views'),
                                                first('page_id').alias('page_id'))

    # 4. Rank titles
    window = Window.partitionBy('date').orderBy(col("tot_count_views").desc())
    df_agg = df_agg.withColumn("rank", row_number().over(window))

    return df_agg


def automated_main():
    save_path = "/scratch/descourt/processed_data_050223"
    os.makedirs(save_path, exist_ok=True)
    save_file = "pageviews_agg_en_2015-2023.parquet"
    project = 'en.wikipedia'

    # Process data
    for args_m, args_y in zip([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                               [7, 8, 9, 10, 11, 12],
                               [1, 2, 3]],

                              [['2016', '2017', '2018', '2019', '2020', '2021', '2022'],
                               ['2015'],
                               ['2023']]):
        months = [str(m) if m / 10 >= 1 else f"0{m}" for m in args_m]
        dates = [f"{year}-{month}" for year in args_y for month in months]

        dfs = setup_data(years=args_y, months=months)

        # For data < 2015-12, page ids are missing, so we match them with closest date dataset page ids
        if '2015' in args_y:
            dfs = match_ids(dfs, '2015-12', project=project)

        df_filt = filter_data(dfs, project, dates=dates)
        df_agg = aggregate_data(df_filt)
        df_agg.write.parquet(os.path.join(save_path, f"pageviews_agg_{project}_{'_'.join(args_y)}.parquet"))

    # Read all again and save
    dfs_path = [os.path.join(save_path, d) for d in os.listdir(save_path)]
    dfs = [spark.read.parquet(df) for df in dfs_path]
    reduce(DataFrame.unionAll, dfs).write.parquet(os.path.join(save_path, save_file))


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

def compute_false_positive_post():

    save_path = "/scratch/descourt/processed_data_050223"
    os.makedirs(save_path, exist_ok=True)
    save_file = "pageviews_agg_en_2015-2023_fp.parquet"
    project = 'en.wikipedia'

    # Process data
    dfs_fp = []
    for args_m, args_y in tqdm(zip([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                               [7, 8, 9, 10, 11, 12],
                               [1, 2, 3]],

                              [['2016', '2017', '2018', '2019', '2020', '2021', '2022'],
                               ['2015'],
                               ['2023']])):
        months = [str(m) if m / 10 >= 1 else f"0{m}" for m in args_m]
        dates = [f"{year}-{month}" for year in args_y for month in months]

        dfs = setup_data(years=args_y, months=months)
        df_filt = filter_data(dfs, project, dates=dates, remove_false_pos=False)

        # Take top 100 pages (should be a good estimation of actual ranking irr. of redirects)
        df_agg = df_filt.groupBy('date', 'page').agg(sum('counts').alias('tot_counts')).sort(
            desc('tot_counts')).limit(1000).cache()
        # Compute number of counts for mobile versus web
        df_agg = df_agg.join(df_filt.where('access_type = "desktop"').groupBy('date', 'page').agg(
            sum('counts').alias('tot_counts_desktop')), on=['date', 'page'])
        df_agg = df_agg.join(df_filt.where('access_type = "mobile-web"').groupBy('date', 'page').agg(
            sum('counts').alias('tot_counts_mobweb')), on=['date', 'page'])
        # Get false positive titles based on desktop to mobile web views ratio
        df_FP = df_agg.where((col('tot_counts_desktop') / col('tot_counts_mobweb') >= 1e4) | (
                    col('tot_counts_desktop') / col('tot_counts_mobweb') <= 1 / 1e4)).withColumn('toDelete', lit(1))

        dfs_fp.append(df_FP)

    dfs_fp = reduce(DataFrame.unionAll, dfs_fp)
    dfs_fp.write.parquet(os.path.join(save_path, save_file))


if __name__ == '__main__':
    compute_false_positive_post()
