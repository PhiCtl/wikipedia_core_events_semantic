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
from itertools import chain
import requests

conf = pyspark.SparkConf().setMaster("local[3]").setAll([
    ('spark.driver.memory', '50G'),
    ('spark.executor.memory', '50G'),
    ('spark.driver.maxResultSize', '0'),
    ('spark.executor.cores', '3'),
    ('spark.local.dir', '/scratch/descourt/spark')
])
# create the session
spark = SparkSession.builder.config(conf=conf).getOrCreate()
# create the context
sc = spark.sparkContext
sc.setLogLevel('ERROR')


def chunk_split(list_of_ids, chunk_len=49):
    """
    Split the given list into chunks
    :param : list_of_ids
    :param : chunk_len
    Usage : for Wikipedia API, max 50 approx. simultaneous requests
    """

    l = []
    for i in range(0, len(list_of_ids), chunk_len):
        l.append(list_of_ids[i:i + chunk_len])
    if len(l) * 49 < len(list_of_ids):
        l.append(list_of_ids[i:])
    return l

def yield_mapping(pages):
    """
    Parse API request response to get target page to redirects page ids mapping
    :param pages: part of API request response
    """
    inv_mapping = {}

    # Collect all redirects ids
    for p_id, p in pages.items():
        if 'redirects' not in p:
            inv_mapping[p_id] = [p_id]
        else:
            rids = [str(d['pageid']) for d in p['redirects']]
            inv_mapping[p_id] = rids

    return inv_mapping

def query_target_id(request):

    """
    Query Wikipedia API with specified parameters.
    Adapted From https://github.com/pgilders/WikiNewsNetwork-01-WikiNewsTopics
    Parameters
    ----------
    request : dict
        API call parameters.
    Raises
    ------
    ValueError
        Raises error if returned by API.
    Yields
    ------
    dict
        Subsequent dicts of json API response.
    """

    lastContinue = {}
    while True:
        # Clone original request
        req = request.copy()
        # Modify with values from the 'continue' section of the last result.
        req.update(lastContinue)
        # Call API
        result = requests.get(
            'https://en.wikipedia.org/w/api.php', params=req).json()
        if 'error' in result:
            print('ERROR')
            raise ValueError(result['error'])
        if 'warnings' in result:
            print(result['warnings'])
        if 'query' in result:
            yield result['query']['pages']
        if 'continue' not in result:
            break
        lastContinue = result['continue']

def get_target_id(redirect_ids):

    """
    Map the redirect pages ids to their target page id
    :param redirect_ids: list of redirect_ids
    """

    chunk_list = chunk_split(redirect_ids)
    print(f"Matching {len(redirect_ids)} ids")
    mapping = {}

    for chunk in tqdm(chunk_list):
        params = {'action': 'query', 'format': 'json', 'pageids': '|'.join(chunk),
                  'redirects': 'True', 'prop': 'redirects', 'rdlimit': 'max'}
        for res in query_target_id(params):
            m = yield_mapping(res)
            for k in m.keys():
                if k in mapping:
                    mapping[k].extend(m[k])
                else:
                    mapping[k] = m[k]
    return invert_mapping(mapping, redirect_ids)

def invert_mapping(inv_map, redirect_ids):

    """
    Invert mapping and select relevant keys
    """
    mapping = {v : k for k, vs in inv_map.items() for v in vs if v in redirect_ids}
    return mapping



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
        return ['main_page', '-', 'search']
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
        .select(lower(col('page')).alias('page'), col('counts').cast('float'), 'date', 'page_id', 'access_type')
    df_filt = df_filt.filter(~df_filt.page.contains('user:') & \
                             ~df_filt.page.contains('wikipedia:') & \
                             ~df_filt.page.contains('file:') & \
                             ~df_filt.page.contains('mediawiki:') & \
                             ~df_filt.page.contains('template:') & \
                             ~df_filt.page.contains('help:') & \
                             ~df_filt.page.contains('category:') & \
                             ~df_filt.page.contains('portal:') & \
                             ~df_filt.page.contains('draft:') & \
                             ~df_filt.page.contains('timedtext:') & \
                             ~df_filt.page.contains('module:') & \
                             ~df_filt.page.contains('special:') & \
                             ~df_filt.page.contains('media:') & \
                             ~df_filt.page.contains('talk:') & \
                             ~df_filt.page.isin(specials_to_filt) \
                             & (df_filt.counts >= 1))

    # if remove_false_pos:
    #     df_filt = collect_false_positive(df_filt)

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


def collect_false_positive(df_to_clean, lim=200, thresh=1e3):

    """
    Return data
    """
    # Take top 100 pages (should be a good estimation of actual ranking irr. of redirects)
    df_agg = df_to_clean.groupBy('date', 'page').agg(sum('counts').alias('tot_counts')).sort(desc('tot_counts')).limit(lim).cache()
    # Compute number of counts for mobile versus web
    df_agg = df_agg.join(df_to_clean.where('access_type = "desktop"').groupBy('date', 'page').agg(sum('counts').alias('tot_counts_desktop')), on=['date', 'page'])
    df_agg = df_agg.join(df_to_clean.where('access_type = "mobile-web" OR access_type = "mobile-app"')\
                                    .groupBy('date', 'page')\
                                    .agg(sum('counts').alias('tot_counts_mobweb')), on=['date', 'page'])
    # Get false positive titles based on desktop to mobile web views ratio
    df_FP = df_agg.where((col('tot_counts_desktop') / col('tot_counts_mobweb') >= thresh) | (col('tot_counts_desktop') / col('tot_counts_mobweb') <= 1 / thresh)).withColumn('toDelete', lit(1))

    return df_to_clean.join(df_FP.select('date', 'page', 'toDelete'), on=['page', 'date'], how='left').where(col('toDelete').isNull())


def aggregate_data(df, match_ids=True, match_ids_per_access_type=False):
    """
    Compute the aggregated number of views for each page per month,
    Filter out views without page id
    Compute page ranking per month according to total aggregated page views
    """
    # Just to make sure we don't have two processing steps
    assert(((not match_ids) & (not match_ids_per_access_type)) ^ (match_ids ^ match_ids_per_access_type))

    if match_ids_per_access_type:

        # 1. Sorting by descending aggregated counts and grouping by page, select first page id
        # = get main page id for all linking redirects
        w = Window.partitionBy('date', 'page').orderBy(col("tot_count_views").desc())
        df_agg = df.groupBy('date', 'page', 'page_id', 'access_type') \
            .agg(sum('counts').alias('tot_count_views')) \
            .withColumn('page_id', first('page_id').over(w)).cache()

        # 2. Sorting by descending aggregated counts and grouping by page id, select first page title
        # = get main page titles for all page ids
        w = Window.partitionBy('date', 'page_id', 'access_type').orderBy(col("tot_count_views").desc())
        df_agg = df_agg.withColumn('page', first('page').over(w)).cache()
        # Remove annoying pages where the page id is null
        df_agg = df_agg.where(~col('page_id').isNull() & ~(df_agg.page_id == 'null'))

        # 3. SUm by access type, page, page id, date
        df_agg = df_agg.groupBy('date', 'page', 'page_id', 'access_type').agg(
            sum('tot_count_views').alias('tot_count_views')).cache()

        # Gather info
        df_agg_D = df_agg.where('access_type = "desktop"').groupBy('date', 'page', 'page_id').agg(
            sum('tot_count_views').alias('desktop_views')).cache()
        df_agg_M = df_agg.where(df_agg.access_type.contains('mobile')).groupBy('date', 'page', 'page_id').agg(
            sum('tot_count_views').alias('mobile_views')).cache()
        df_agg_f = df_agg_D.join(df_agg_M, on=['date', 'page', 'page_id'], how='full_outer').cache()

        return df_agg_f

    if match_ids:

        # 1. Aggregate counts by page title and page ids &
        #    Sorting by descending aggregated counts and grouping by page, select first page id
        # = Get main page id for all redirects
        w = Window.partitionBy('date', 'page').orderBy(col("tot_count_views").desc())
        df_agg = df.groupBy('date', 'page', 'page_id') \
            .agg(sum('counts').alias('tot_count_views')) \
            .withColumn('page_id', first('page_id').over(w))

        # 2. Sorting by descending aggregated counts and grouping by page id, select first page title
        # = Gather all redirects and main page counts
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

def match_missing_ids():

    print('Load data')
    dfs = spark.read.parquet("/scratch/descourt/processed_data_050923/pageviews_en_2015-2023.parquet")
    df_topics_sp = spark.read.parquet('/scratch/descourt/topics/topics-enwiki-20230320-parsed.parquet')

    print('Merge with topics and check what matches')
    dfs_topics_merged = dfs.drop('rank', 'fractional_rank')\
                           .where((dfs.page_id != 'null') & col('page_id').isNotNull())\
                           .join(df_topics_sp.select('page_id', 'topics_unique').distinct(), 'page_id', 'left')
    dfs_matched = dfs_topics_merged.where(col('topics_unique').isNotNull()).drop('topics_unique').cache()
    dfs_unmatched = dfs_topics_merged.where(col('topics_unique').isNull()).drop('topics_unique').cache()

    unmatched_count, matched_count = dfs_unmatched.count(), dfs_matched.count()
    print(f"Tot count {dfs.count()}, unmatched counts {unmatched_count} matched counts {matched_count} sum {unmatched_count + matched_count}")

    print('Matching the unmatched ids')
    unmatched_ids = [i['page_id'] for i in dfs_unmatched.select('page_id').distinct().collect()]
    mappings = get_target_id(unmatched_ids)
    mapping_expr = create_map([lit(x) for x in chain(*mappings.items())])
    dfs_unmatched = dfs_unmatched.withColumn('matched_page_id', mapping_expr[col("page_id")]).drop('page_id')
    dfs_unmatched = dfs_unmatched.join(df_topics_sp.select('page_id', 'topics_unique').distinct(),
                                             dfs_unmatched.matched_page_id == df_topics_sp.page_id, 'left').cache()
    print(f"New match count in unmatched {dfs_unmatched.where(col('topics_unique').isNotNull()).count()}")

    dfs_final_matched = dfs_unmatched.select('page', col('matched_page_id').alias('page_id'), 'tot_count_views', 'date').union(
        dfs_matched.select('page', 'page_id', 'tot_count_views', 'date'))

    print("Match the pages with the ids and recompute the tot view counts")
    w = Window.partitionBy('date', 'page_id').orderBy(desc('tot_count_views'))
    dfs_final_matched = dfs_final_matched.withColumn('page', first('page').over(w))
    dfs_final_matched = dfs_final_matched.groupBy('date', 'page_id', 'page').agg(sum('tot_count_views').alias('tot_count_views'))

    print("Compute ranks")
    window = Window.partitionBy('date').orderBy(col("tot_count_views").desc())
    dfs_final_matched = dfs_final_matched.withColumn("rank", row_number().over(window))
    df_fract = dfs_final_matched.groupBy('date', 'tot_count_views').agg(avg('rank').alias('fractional_rank'))
    dfs_final_matched = dfs_final_matched.join(df_fract, on=['date', 'tot_count_views'])

    print("Write to file")
    dfs_final_matched.write.parquet("/scratch/descourt/processed_data_050923/pageviews_en_2015-2023_matched.parquet")

def download_mappings():

    # TODO merge with above code for the sake of reproducibility

    print('Load redirect_ids')
    with open("/scratch/descourt/topics/mappings_ids.pickle", "rb") as handle:
        old_wrong_mapping = pickle.load(handle)

    redirect_ids = [i for i in old_wrong_mapping.keys()]

    print('Matching the unmatched ids')
    mappings = get_target_id(redirect_ids)
    with open("/scratch/descourt/topics/mappings_ids_corrected.pickle", "wb") as handle:
        pickle.dump(mappings, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Saving file as parquet for further matching")
    mappings_spark = [(k, v) for k, v in mappings.items()]
    df_matching = spark.createDataFrame(data=mappings_spark, schema=["redirect", "target"])
    df_matching.write.parquet("/scratch/descourt/topics/df_missing_redirects.parquet")

    print("Done")


if __name__ == '__main__':
    download_mappings()
