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

from src.redirects_helpers import *


def setup_data(years, months, spark_session, path="/scratch/descourt/raw_data/pageviews"):
    """
    Load and prepare wikipedia projects pageviews data for given year and month
    :return pyspark dataframe
    """

    def read_file(f_n, date):
        print(f"loading {f_n}")
        df = spark_session.read.csv(f_n, sep=r' ')
        return df.selectExpr("_c0 as project", "_c1 as page", "_c2 as page_id", "_c3 as access_type", "_c4 as counts",
                             "_c5 as anonym_user")\
            .withColumn('date', lit(date))

    files_names = [os.path.join(path, f"pageviews-{year}{month}-user.bz2") for year in years for month in months]
    dates = [f"{year}-{month}" for year in years for month in months]
    start = time.time()
    dfs = [read_file(f, d) for f, d in zip(files_names, dates)]
    df = reduce(DataFrame.unionAll, dfs)
    print(f"Elapsed time {time.time() - start} s")
    return df


def specials(project):
    """
    Filter out some special pages
    """
    if project == 'en':
        return ['Main_Page', '-', 'Search']
    else :
        return ['-']


def filter_data(df, projects, dates):
    """
    Filter in wanted data from initial dataframe
    Based on a list of pages to exclude, on the dates we want to include and on the edition we want to study
    :param df: dataframe (pyspark)
    :param projects: eg. [en, fr]
    :param dates: list of strings of format YYYY-MM
    """

    specials_to_filt = ['Main_Page', '-', 'Search']
    df_filt = df.where(df.project.isin([l+'.wikipedia' for l in projects])) \
        .filter(df.date.isin(dates)) \
        .select(col('page').alias('page'), col('counts').cast('float'), 'date', 'page_id', 'access_type',
                split('project',  '\.')[0].alias('project') )
    if 'en' in projects and len(projects) == 1:
        df_filt = df_filt.filter(~df_filt.page.contains('User:') & \
                                 ~df_filt.page.contains('Wikipedia:') & \
                                 ~df_filt.page.contains('File:') & \
                                 ~df_filt.page.contains('MediaWiki:') & \
                                 ~df_filt.page.contains('Template:') & \
                                 ~df_filt.page.contains('Help:') & \
                                 ~df_filt.page.contains('Category:') & \
                                 ~df_filt.page.contains('Portal:') & \
                                 ~df_filt.page.contains('Draft:') & \
                                 ~df_filt.page.contains('TimedText:') & \
                                 ~df_filt.page.contains('Module:') & \
                                 ~df_filt.page.contains('Special:') & \
                                 ~df_filt.page.contains('Media:') & \
                                 ~df_filt.page.contains('Talk:') & \
                                 ~df_filt.page.contains('talk:') & \
                                 ~df_filt.page.isin(specials_to_filt) \
                                 & (df_filt.counts >= 1))
        return df_filt
    elif 'fr' in projects and len(projects) == 1:
        df_filt = df_filt.filter(~df_filt.page.contains('Utilisateur:') & \
                                 ~df_filt.page.contains('Wikipédia:') & \
                                 ~df_filt.page.contains('Fichier:') & \
                                 ~df_filt.page.contains('MediaWiki:') & \
                                 ~df_filt.page.contains('Modèle:') & \
                                 ~df_filt.page.contains('Aide:') & \
                                 ~df_filt.page.contains('Catégorie:') & \
                                 ~df_filt.page.contains('Portail:') & \
                                 ~df_filt.page.contains('Projet:') & \
                                 ~df_filt.page.contains('TimedText') & \
                                 ~df_filt.page.contains('Référence:') & \
                                 ~df_filt.page.contains('Module:') & \
                                 ~df_filt.page.contains('Gadget:') & \
                                 ~df_filt.page.contains('Sujet:') & \
                                 ~df_filt.page.contains('Discussion') & \
                                 ~df_filt.page.contains('Spécial') & \
                                 ~df_filt.page.isin(specials_to_filt) \
                                 & (df_filt.counts >= 1))
        return df_filt
    else :
        df_filt = df_filt.where(~df_filt.page.contains(':') & ~df_filt.page.isin(specials_to_filt)\
                                & (df_filt.counts >= 1))

    return df_filt


def match_ids(df, latest_date, projects):
    """
    Match page ids from latest date dataframe with pageids
    Especially for data before 2015-12 because page ids weren't matched
    """

    [y, m] = latest_date.split('-')

    # Select all available pages and their page ids (raw) for project of interest
    df_latest = setup_data([y], [m], path=f'/scratch/descourt/raw_data/pageviews', spark_session=spark)  # Download all data

    # Select columns of interest and filter project
    df_latest = df_latest.where((df_latest.project.isin([f"{p}.wikipedia" for p in projects]))\
                                 & (df_latest.page_id != 'null')) \
        .select('page', 'page_id', 'counts', 'project')

    # Select unique id per page
    w = Window.partitionBy('project','page').orderBy(col("tot_count_views").desc())
    df_pageids = df_latest.groupBy('project', 'page', 'page_id') \
        .agg(sum('counts').alias('tot_count_views')) \
        .withColumn('page_id', first('page_id').over(w)) \
        .select('project', 'page', 'page_id').distinct().cache()

    # Join on page title to recover page ids if any
    df = df.drop('page_id')\
          .where(df.project.isin([f"{p}.wikipedia" for p in projects]))\
          .join(df_pageids, on=['page', 'project'], how='left')

    return df


def aggregate_data(df, match_ids=True, match_ids_per_access_type=False):
    """
    Compute the aggregated number of views for each page per month,
    Filter out views with "Null" page id
    Compute page ranking per month according to total aggregated page views

    :param df
    :param match_ids: whether to compute aggregated views based on page_id (if not, based on page title)
    :param match_ids_per_access_type : whether to compute aggregated views based on page_id and access_type
    """
    # Just to make sure we don't have two processing steps
    assert (((not match_ids) & (not match_ids_per_access_type)) ^ (match_ids ^ match_ids_per_access_type))

    if match_ids_per_access_type:
        # 1. Sorting by descending aggregated counts and grouping by page, select first page id
        # = get main page id for all linking redirects
        w = Window.partitionBy('project', 'date', 'page').orderBy(col("tot_count_views").desc())
        df_agg = df.groupBy('project', 'date', 'page', 'page_id', 'access_type') \
            .agg(sum('counts').alias('tot_count_views')) \
            .withColumn('page_id', first('page_id').over(w)).cache()

        # 2. Sorting by descending aggregated counts and grouping by page id, select first page title
        # = get main page titles for all page ids
        w = Window.partitionBy('project', 'date', 'page_id', 'access_type').orderBy(col("tot_count_views").desc())
        df_agg = df_agg.withColumn('page', first('page').over(w)).cache()
        # Remove annoying pages where the page id is null
        df_agg = df_agg.where(~col('page_id').isNull() & ~(df_agg.page_id == 'null'))

        # 3. SUm by access type, page, page id, date
        df_agg = df_agg.groupBy('project', 'date', 'page', 'page_id', 'access_type').agg(
            sum('tot_count_views').alias('tot_count_views')).cache()

        # Gather info
        df_agg_D = df_agg.where('access_type = "desktop"').groupBy('project', 'date', 'page', 'page_id').agg(
            sum('tot_count_views').alias('desktop_views')).cache()
        df_agg_M = df_agg.where(df_agg.access_type.contains('mobile')).groupBy('project', 'date', 'page', 'page_id').agg(
            sum('tot_count_views').alias('mobile_views')).cache()
        df_agg_f = df_agg_D.join(df_agg_M, on=['date', 'page', 'page_id', 'project'], how='full_outer').cache()

        return df_agg_f

    if match_ids:

        # 1. Aggregate counts by page title and page ids &
        #    Sorting by descending aggregated counts and grouping by page, select first page id
        # = Get main page id for all redirects
        w = Window.partitionBy('project', 'date', 'page').orderBy(col("tot_count_views").desc())
        df_agg = df.groupBy('project', 'date', 'page', 'page_id') \
            .agg(sum('counts').alias('tot_count_views')) \
            .withColumn('page_id', first('page_id').over(w))

        # 2. Sorting by descending aggregated counts and grouping by page id, select first page title
        # = Gather all redirects and main page counts
        w = Window.partitionBy('project', 'date', 'page_id').orderBy(col("tot_count_views").desc())
        df_agg = df_agg.withColumn('page', first('page').over(w))

        # 3. Aggregate counts by page title
        df_agg = df_agg.groupBy('project', 'date', 'page').agg(sum('tot_count_views').alias('tot_count_views'),
                                                    first('page_id').alias('page_id'))
        df_agg = df_agg.where(~col('page_id').isNull() & ~(df_agg.page_id == 'null'))
    else:
        df_agg = df.groupBy('project', 'date', 'page').agg(sum("counts").alias('tot_count_views'),
                                                first('page_id').alias('page_id'))

    # 4. Rank titles
    window = Window.partitionBy('project', 'date').orderBy(col("tot_count_views").desc())
    df_agg = df_agg.withColumn("rank", row_number().over(window))

    return df_agg


def automated_main():
    save_path = "/scratch/descourt/processed_data/fr"
    os.makedirs(save_path, exist_ok=True)
    save_file = "pageviews_fr_2015-2023.parquet"

    # Process data
    for args_m, args_y in zip([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                               [7, 8, 9, 10, 11, 12],
                               [1, 2, 3]],

                              [['2016', '2017', '2018', '2019', '2020', '2021', '2022'],
                               ['2015'],
                               ['2023']]):
        months = [str(m) if m / 10 >= 1 else f"0{m}" for m in args_m]
        dates = [f"{year}-{month}" for year in args_y for month in months]

        dfs = setup_data(years=args_y, months=months, spark_session=spark, path="/scratch/descourt/raw_data/pageviews")

        # For data < 2015-12, page ids are missing, so we match them with closest date dataset page ids
        if '2015' in args_y:
            dfs = match_ids(dfs, '2015-12', projects=['fr'])

        df_filt = filter_data(dfs, ['fr'], dates=dates)
        df_agg = aggregate_data(df_filt)
        df_agg.write.parquet(os.path.join(save_path, f"pageviews_agg_{'-'.join(['fr'])}_{'_'.join(args_y)}.parquet"))

    # Read all again and save
    dfs_path = [os.path.join(save_path, d) for d in os.listdir(save_path)]
    dfs = [spark.read.parquet(df) for df in dfs_path]
    reduce(DataFrame.unionAll, dfs).write.parquet(os.path.join(save_path, save_file))


def match_missing_ids(dfs=None, df_topics_sp=None, save_interm=True):

    """
    Further matching is needed between redirects and target pages in some cases
    Here we make use of the topic-page dataset which encloses all the possible target page ids,
    and check when we cannot match a topic to a page based on its id
    Either this id is a page which didn't exist at the time the topic-page dataset was extracted,
    either this is a redirect id

    We then query Wikipedia's API to match the redirect page id with the target page id
    """

    parser = argparse.ArgumentParser(
        description='Wikipedia missing pageids downloading')
    parser.add_argument('--year',
                        type=str,
                        default='2020',
                        help='Year to download')
    parser.add_argument('--project',
                        type=str,
                        default='en')
    args = parser.parse_args()
    year = args.year
    project = args.project

    print('Load data')
    # TODO make below project specific -> move processed data file from to _en because for french it has _fr
    dfs = spark.read.parquet("/scratch/descourt/processed_data/en/pageviews_en_2015-2023.parquet")
    # TODO remove years
    dfs_2019 = dfs.where(dfs.date.contains('2019'))
    # TODO Make project specific
    if df_topics_sp is None:
        df_topics_sp = spark.read.parquet('/scratch/descourt/metadata/topics/topic_en/topics-enwiki-20230320-parsed.parquet')

    # print('Merge with topics and retrieve which page_ids do not match')
    # df_unmatched = dfs_2019.where((dfs_2019.page_id != 'null') & col('page_id').isNotNull()) \
    #     .join(df_topics_sp.select('page_id', 'topics_unique').distinct(), 'page_id', 'left')\
    #     .where(col('topics_unique').isNull()).select('page_id').distinct()
    # unmatched_ids = [str(p['page_id']) for p in df_unmatched.select('page_id').collect()]
    #
    # print('Match the unmatched ids with their target page id')
    # # TODO make proejct specific
    # mappings = get_target_id(unmatched_ids, project='en')
    # if save_interm:
    #     with open(f"/scratch/descourt/topics/topic_en/mappings_ids_corrected_2019.pickle", "wb") as handle:
    #         pickle.dump(mappings, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # mappings_spark = [(k, v) for k, v in mappings.items()]
    with open(f"/scratch/descourt/metadata/topics/topic_en/mappings_ids_corrected_2019.pickle", "rb") as handle:
        mappings = pickle.load(handle)

    mappings_spark = [(k, v) for k, v in mappings.items()]
    df_matching = spark.createDataFrame(data=mappings_spark, schema=["redirect", "target"])
    if save_interm:
        df_matching.write.parquet(f"/scratch/descourt/metadata/topics/topic_en/df_missing_redirects_2019.parquet")

    dfs_2019 = dfs_2019.join(df_matching, dfs_2019.page_id == df_matching.redirect, 'left')
    # The left unmatched page_ids correspond in fact already to target pages,
    # so their own id could not be matched and we replace it with original id
    dfs_2019 = dfs_2019.withColumn('page_id', coalesce('target', 'page_id'))

    print("Recompute the tot view counts")
    w = Window.partitionBy('date', 'page_id').orderBy(desc('tot_count_views'))
    dfs_2019 = dfs_2019.withColumn('page', first('page').over(w))
    dfs_2019 = dfs_2019.groupBy('date', 'page_id', 'page').agg(
        sum('tot_count_views').alias('tot_count_views'))

    print("Recompute ordinal and fractional ranks")
    window = Window.partitionBy('date').orderBy(col("tot_count_views").desc())
    dfs_2019 = dfs_2019.withColumn("rank", row_number().over(window))
    df_fract = dfs_2019.groupBy('date', 'tot_count_views').agg(avg('rank').alias('fractional_rank'))
    dfs_2019 = dfs_2019.join(df_fract, on=['date', 'tot_count_views'])

    print("Write to file")
    dfs = dfs.where(~dfs.date.contains('2019')).union(dfs_2019)
    # TODO make project specific
    dfs.write.parquet("/scratch/descourt/processed_data/en/pageviews_en_2015-2023_matched.parquet")

    print("Done")

def match_over_months():
    """
    Match page ids and names over months so as to keep track of articles over time
    """

    dfs = spark.read.parquet("/scratch/descourt/processed_data/fr/pageviews_fr_2015-2023.parquet")
    dfs = dfs.withColumn('date', to_date(col('date'),'yyyy-MM')).where(col('date') <= to_date(lit('2022-11'), 'yyyy-MM')).cache()

    w_asc = Window.partitionBy('page_id').orderBy(asc(col('date')))
    w_desc = Window.partitionBy('page_id').orderBy(desc(col('date')))
    dfs_change = dfs.select(col('page_id').alias('page_id_0'),
                            first('page').over(w_asc).alias('first_name'),
                            first('page').over(w_desc).alias('last_name'),
                            first(col('date')).over(w_asc).alias('first_date'),
                            add_months(first(col('date')).over(w_desc), 1).alias('last_date')) \
                    .distinct().cache()

    n, i = 10, 1
    while n > 0 and i <= 10:
        print(f"{i} - {n}")
        dfs_change = dfs_change.alias('a') \
            .join(dfs_change.alias('b'),
                  (col("a.last_name") == col("b.first_name")) & (col("a.last_date") == col("b.first_date")), 'left') \
            .select(col("a.first_name"), col('a.last_name'), col("b.last_name").alias(f'last_name_{i}'),
                    col("a.first_date"), col('a.last_date'), col("b.last_date").alias(f'last_date_{i}'),
                    col(f"b.page_id_{i - 1}").alias(f"page_id_{i}"),
                    *[col(f"a.page_id_{l}") for l in range(i)]).cache()
        dfs_change = dfs_change.withColumn('last_name', coalesce(f'last_name_{i}', 'last_name')) \
            .withColumn('last_date', coalesce(f'last_date_{i}', 'last_date')).drop(f'last_name_{i}',
                                                                                   f'last_date_{i}').cache()

        n = dfs_change.where(col(f"page_id_{i}").isNotNull()).count()
        i += 1

    print(n)
    dfs_change.write.parquet("/scratch/descourt/processed_data/df/pageviews_fr_articles_ev_nov22.parquet")


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
    match_over_months()
