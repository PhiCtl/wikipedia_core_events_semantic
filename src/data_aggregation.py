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

    df_filt = df.where(df.project.isin([l+'.wikipedia' for l in projects])) \
        .filter(df.date.isin(dates)) \
        .select(col('page').alias('page'), col('counts').cast('float'), 'date', 'page_id', 'access_type',
                split('project',  '\.')[0].alias('project') )

    # Filter articles namespace
    df_filt = per_project_filt(df_filt, projects)
    return df_filt


def match_ids(df, latest_date, projects, raw_path=f'/scratch/descourt/raw_data/pageviews'):
    """
    Match page ids from latest date dataframe with pageids
    Especially for data before 2015-12 because page ids are not present at that time
    """

    [y, m] = latest_date.split('-')

    # Select all available pages and their page ids (raw) for project of interest
    df_latest = setup_data([y], [m], path=raw_path, spark_session=spark)  # Download all data

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
        df_agg = df.groupBy('project', 'date', 'page', 'page_id').agg(sum("counts").alias('tot_count_views'))

    # 4. Rank titles
    window = Window.partitionBy('project', 'date').orderBy(col("tot_count_views").desc())
    df_agg = df_agg.withColumn("rank", row_number().over(window))

    return df_agg


def automated_main():

    """Process single edition of Wikipedia pageviews for July 2015-March 2023 period"""

    parser = argparse.ArgumentParser(
        description='Wikipedia monthly pageviews dumps processing from July 2015 up to March 2023')
    parser.add_argument('--project',
                        type=str,
                        default='en')
    args = parser.parse_args()
    project = args.project

    save_path = f"/scratch/descourt/processed_data/{project}"
    os.makedirs(save_path, exist_ok=True)
    save_file = f"pageviews_{project}_2015-2023.parquet"

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
            dfs = match_ids(dfs, '2015-12', projects=[project])

        df_filt = filter_data(dfs, [project], dates=dates)
        df_agg = aggregate_data(df_filt)
        df_agg.write.parquet(os.path.join(save_path, f"pageviews_agg_{'-'.join([project])}_{'_'.join(args_y)}.parquet"))

    # Read all again and save
    dfs_path = [os.path.join(save_path, d) for d in os.listdir(save_path)]
    dfs = [spark.read.parquet(df) for df in dfs_path]
    reduce(DataFrame.unionAll, dfs).write.parquet(os.path.join(save_path, save_file))


def match_missing_ids(save_interm=True):

    """
    Further matching is needed between redirects and target pages in some cases
    Here we make use of the topic-page dataset which encloses all the possible target page ids,
    and check when we cannot match a topic to a page based on its id
    Either this id is a page which didn't exist at the time the topic-page dataset was extracted,
    either this is a redirect id

    TODO We didn't take into account the fact that pages could be moved here, and it has to be corrected

    We then query Wikipedia's API to match the redirect page id with the target page id
    """

    parser = argparse.ArgumentParser(
        description='Wikipedia missing pageids downloading')

    parser.add_argument('--project',
                        type=str,
                        default='en',
                        choices=['en', 'fr'])
    args = parser.parse_args()
    project = args.project

    print('Load data')
    dfs = spark.read.parquet(f"/scratch/descourt/processed_data/{project}/pageviews_{project}_2015-2023.parquet")
    if project == 'en':
        df_topics_sp = spark.read.parquet('/scratch/descourt/metadata/topics/topic_en/topics-enwiki-20230320-parsed.parquet')\
            .select('page_id', col('topics_specific_unique').alias('topic')).distinct().cache()
    else:
        # TODO not optimal for other editions as we have data points up to March 23
        df_topics_sp = spark.read.parquet(f'/scratch/descourt/metadata/akhils_data/wiki_nodes_topics_2022-09_{project}.parquet')

    print('Merge with topics and retrieve which page_ids do not match')
    df_unmatched = dfs.where((dfs.page_id != 'null') & col('page_id').isNotNull()) \
        .join(df_topics_sp.select('page_id', 'topic').distinct(), 'page_id', 'left')\
        .where(col('topic').isNull()).select('page_id').distinct()
    unmatched_ids = [str(p['page_id']) for p in df_unmatched.select('page_id').collect()]

    print('Match the unmatched ids with their target page id')
    mappings = get_target_id(unmatched_ids, project=project)

    mappings_spark = [(k, v) for k, v in mappings.items()]
    df_matching = spark.createDataFrame(data=mappings_spark, schema=["redirect", "target"])
    if save_interm:
        df_matching.write.parquet(f"/scratch/descourt/metadata/redirect_matching/df_missing_redirects_{project}.parquet")

    dfs = dfs.join(df_matching, dfs.page_id == df_matching.redirect, 'left')
    # The left unmatched page_ids correspond in fact already to target pages,
    # so their own id could not be matched and we replace it with original id
    dfs = dfs.withColumn('page_id', coalesce('target', 'page_id'))

    print("Recompute the tot view counts")
    w = Window.partitionBy('date', 'page_id').orderBy(desc('tot_count_views'))
    dfs = dfs.withColumn('page', first('page').over(w))
    dfs = dfs.groupBy('date', 'page_id', 'page').agg(
        sum('tot_count_views').alias('tot_count_views'))

    print("Recompute ordinal and fractional ranks")
    window = Window.partitionBy('date').orderBy(col("tot_count_views").desc())
    dfs = dfs.withColumn("rank", row_number().over(window))
    df_fract = fs.groupBy('date', 'tot_count_views').agg(avg('rank').alias('fractional_rank'))
    dfs = dfs.join(df_fract, on=['date', 'tot_count_views'])

    print("Write to file")
    dfs.write.parquet(f"/scratch/descourt/processed_data/{project}/pageviews_{project}_2015-2023_matched.parquet")

    print("Done")

def match_over_months():
    """
    Match page ids over months based on names so as to keep track of articles over time
    It enables correction of some page moves by article's name and page id tracking to a certain extent
    Indeed, pages which are not viewed during a given month are not kept in our dataset, hence appear as missing
    and might pop up again the following month

    Note that it represents a very tiny fraction of the data
    """

    parser = argparse.ArgumentParser(
        description='Wikipedia page ids and page name matching over time')

    parser.add_argument('--project',
                        type=str,
                        default='en',
                        choices=['en', 'fr'])
    parser.add_argument('--date',
                        type=str,
                        default='2023-03',
                        help='date up to which we want to track articles')
    args = parser.parse_args()
    project = args.project
    date = args.date

    # Select data up to date
    dfs = spark.read.parquet(f"/scratch/descourt/processed_data/{project}/pageviews_{project}_2015-2023.parquet")
    dfs = dfs.withColumn('date', to_date(col('date'),'yyyy-MM')).where(col('date') <= to_date(lit(date), 'yyyy-MM')).cache()

    w_asc = Window.partitionBy('page_id').orderBy(asc(col('date')))
    w_desc = Window.partitionBy('page_id').orderBy(desc(col('date')))
    dfs_change = dfs.select(col('page_id').alias('page_id_0'),
                            first('page').over(w_asc).alias('first_name'),
                            first('page').over(w_desc).alias('last_name'),
                            first(col('date')).over(w_asc).alias('first_date'),
                            add_months(first(col('date')).over(w_desc), 1).alias('last_date')) \
                    .distinct().cache()

    n, i = 10, 1
    # Retrieve for each article ever seen in the entire Wikipedia edition up to "date"
    # 1. its first name, corresponding page id and the first date at which it appeared in the volume
    # 2. its last name, corresponding page id and the last date at which it appeared in the volume
    # Match on last name if first name at following date is similar
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
    w = Window.partitionBy('last_name', 'last_date').orderBy(desc('first_date'))
    w_asc = Window.partitionBy('last_name', 'last_date').orderBy(asc('first_date'))

    dfs_change_filt = dfs_change.select('last_name', 'last_date',
                                        first('page_id_0').over(w).alias('last_page_id'),
                                        first('first_date').over(w_asc).alias('first_date'),
                                        explode(array(*['page_id_0', 'page_id_1', 'page_id_2', 'page_id_3'])).alias('page_ids')) \
        .dropna(subset=['page_ids']) \
        .dropDuplicates(['last_page_id', 'last_name', 'last_date', 'page_ids']).cache()
    dfs_change_filt.write.parquet(f"/scratch/descourt/processed_data/{project}/pageviews_{project}_articles_ev_{date}.parquet")


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

    # TODO change line below
    match_over_months()
    # match_missing_ids(False)
    # automated_main()
