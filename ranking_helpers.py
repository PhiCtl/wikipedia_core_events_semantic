from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from functools import reduce

from pyspark.sql import *
from pyspark.sql.functions import *
import pyspark

from pyspark.sql.window import Window
from pyspark.ml.feature import QuantileDiscretizer
from pyspark.ml.feature import Bucketizer

def merge_outer(df1, df2):
    return pd.merge(df1, df2, how='outer')
def merge_index(df1, df2):
    return pd.merge(df1, df2, left_index=True, right_index=True)


def compute_ranks_bins(df, lim=100000, slicing=5000, subset='date', log=False):
    """
    Bin ranking into rank ranges, eg. top 0 to 5K -> bin 0, top 5k to 10k -> bin 1, etc..
    :param df : pyspark dataFrame
    :param lim: fixed number of items to retrieve, eg. top 100K pageviews
    :param slicing: discretization of rankings into slices, eg. top 0 - 5000 -> bin 0
    :param subset: on which time subset to group for total views count aggregation (can be date_range)
    """

    # Top lim rank pages for each date range
    window = Window.partitionBy(subset).orderBy(col("tot_count_views").desc())
    df_lim = df.withColumn("rank", row_number().over(window))
    df_lim = df_lim.where(col("rank") <= lim)

    # Bin ranges
    sp = [i for i in range(0, lim, slicing)] + [lim]
    if log:
        sp = [np.power(10, i) for i in range(10)]
    bucketizer = Bucketizer(splits=sp, inputCol="rank", outputCol="rank_range")
    df_buck = bucketizer.transform(df_lim)

    return df_buck

def compute_ranks(df, lim, subset='date'):
    # Top lim rank pages for each date range
    window = Window.partitionBy(subset).orderBy(col("tot_count_views").desc())
    df_lim = df.withColumn("rank", row_number().over(window))
    df_lim = df_lim.where(col("rank") <= lim)

    return df_lim


def compute_freq_bins(df, lim=100000, nb_bins=3, subset='date', max_views=10000000, min_views=5000):
    """
    Bin tot_count_views into rank ranges, eg.
    """

    # Top lim rank pages for each date range
    window = Window.partitionBy(subset).orderBy(col("tot_count_views").desc())
    df_lim = df.withColumn("rank", row_number().over(window))
    df_lim = df_lim.where(col("rank") <= lim)

    # Bin ranges
    views_bins = [i for i in range(min_views, max_views, int((max_views - min_views) / nb_bins))]
    bucketizer = Bucketizer(splits=views_bins + [float("inf")], inputCol="tot_count_views", outputCol="freq_bins")
    df_buck = bucketizer.transform(df_lim)

    return df_buck


def compute_consecutive_bins(df, lim=100000, slicing=5000, nb_bin_dates=3):
    """
    Bin dates into dates ranges, eg. 2020-{01, 02, 03} -> bin 0, 2020-{04, 05, 06} -> bin 1, etc..
    :param df : pyspark dataFrame
    :param lim: fixed number of items to retrieve, eg. top 100K pageviews
    :param slicing: discretization of rankings into slices, eg. top 0 - 5000 -> bin 0
    :param nb_bin_dates : number of dates categories we want to bin data into
    """

    # Bin dates
    # Convert string dates
    df_buck = df.withColumn("datetime", to_date("date", "yyyy-MM")).withColumn("date_int", unix_timestamp("datetime"))
    dates = sorted([d["date_int"] for d in df_buck.select("date_int").distinct().cache().collect()])
    idx = int(len(dates) / nb_bin_dates)
    bucketizer = Bucketizer(splits=dates[::idx] + [dates[-1]], inputCol="date_int", outputCol="date_range")
    df_buck = bucketizer.transform(df_buck)

    # For each bin, compute rank bins
    df_buck = compute_ranks_bins(df_buck, lim=lim, subset='date_range', slicing=slicing)

    return df_buck


def compute_overlaps(df, offset=1, slicing=5000):
    """
    Compute overlaps of pages in given rank range between two dates bins
    Eg. overlap of pages in top 0 to 5K between period
    :param offset: in months. Eg. offset of 2 means we'll compute intersection between Jan and Mar, Feb and Apr, Mar and May, ...
    :param slicing: rank size of slices, see compute_consecutive_bins
    """

    # Group by date and rank bins set of pages
    df_sets = df.groupBy("date_range", "rank_range") \
        .agg(collect_set("page").alias("ranked_pages"))

    # Store comparison pages set
    df_consecutive_sets = df_sets.withColumn("prev_ranked_pages", lag("ranked_pages", offset=offset).over(
        Window.partitionBy("rank_range").orderBy("date_range"))) \
        .dropna(subset="prev_ranked_pages")
    # Compute overlap
    # overlap= udf(lambda r: len(r.ranked_pages.intersection(r.prev_ranked_pages)) / len(r.ranked_pages))
    # df_overlaps = df_consecutive_sets.withColumn("overlap", overlap(struct([df_consecutive_sets[x] for x in df_consecutive_sets.columns])))
    df_overlaps = df_consecutive_sets.select("date_range", "rank_range", (
                size(array_intersect("ranked_pages", "prev_ranked_pages")) / slicing * 100).alias("overlap"))

    return df_overlaps

def compute_merged_overlaps(df, offsets, slicing):
    """
    Compute overlaps of pages in given rank range between two dates bins
    Eg. overlap of pages in top 0 to 5K between period
    :param offset: in months. Eg. offsets of [1,2] means we'll compute intersection between Jan Feb March top pages content
    :param slicing: rank size of slices, see compute_consecutive_bins
    Is computationally intensive I guess
    """

    # Aggregate page views per rank range and date
    df_sets = df.groupBy("date_range", "rank_range") \
        .agg(collect_set("page").alias("ranked_pages"))

    # Store comparison pages set
    for offset in offsets:
        df_sets = df_sets.withColumn(f"prev_ranked_pages_{offset}", lag("ranked_pages", offset=offset).over(
            Window.partitionBy("rank_range").orderBy("date_range")))
    # Store results
    df_overlaps = df_sets.withColumn("overlap", col("ranked_pages"))

    # Compute intersections
    cols = [f"prev_ranked_pages_{offset}" for offset in offsets]
    for col_ in cols:
        df_overlaps = df_overlaps.withColumn("overlap", array_intersect("overlap", col_))
    # Compute exception
    cols = cols + ["ranked_pages"]
    for col_ in cols:
        df_overlaps = df_overlaps.withColumn(col_, array_except(col_, "overlap"))

    # Compute overlap percentage
    df_overlaps = df_overlaps.withColumn("overlap_size", size("overlap") / slicing * 100).dropna(subset=cols)

    return df_overlaps


def compute_overlap_evolution(df, start_date, end_date, rank_range, slicing=5000):
    """
    For a given date, compute intersection of this date top rank_range pages
    with other prev and following dates top rank_range pages
    :param df: Pyspark dataframe
    :param start_date: date which top rank_range pages will be compared to other months
    :param end_date: actually useless parameter
    :param rank_range: select rank range of interest
    :param slicing: slices sizes
    :return: rolling window (Pyspark dataframe) of given date slice intersection
    """
    # Select rank slices for all dates up to end_date
    df_filt = df.filter(df.date <= end_date).where(df.rank_range == rank_range)

    # Compute pages sets for all dates
    df_sets = df_filt.groupBy("date", "rank_range") \
        .agg(collect_set("page").alias("ranked_pages"))
    slice_of_interest = df_sets.where(df.date == start_date).select("ranked_pages").collect()[0]["ranked_pages"]
    df_sets = df_sets.withColumn("comparison_set", array([lit(x) for x in slice_of_interest]))
    # Compute intersection
    df_overlaps = df_sets.select("date",
                                 "rank_range",
                                 (array_intersect("ranked_pages", "comparison_set")).alias("overlap"))

    df_overlaps = df_overlaps.withColumn("overlap_size", size("overlap") / slicing * 100)
    return df_overlaps


def rank_turbulence_divergence_pd(rks, d1, d2, N1, N2, alpha):
    """
    Compute rank turbulence divergence between date d1 and d2 for pages in rks
    :param rks: dataframe with columns d1 and d2
    :param d1: string date 1 in format YYYY-MM
    :param d2: string date 2 in format YYYY-MM
    :param N1: number of elements at d1
    :param N2: number of elements at d2
    :param alpha: hyper parameter for divergence computation
    """
    N = (alpha + 1) / alpha * (
            ((1.0 / rks.loc[~rks[d1].isnull(), d1] ** alpha - 1 / (N1 + 0.5 * N2) ** alpha).abs() ** (1 / (alpha + 1))).sum()
          + ((-1.0 / rks.loc[~rks[d2].isnull(), d2] ** alpha + 1 / (N2 + 0.5 * N1) ** alpha).abs() ** (1 / (alpha + 1))).sum())

    rks[f'div_{d2}'] = (alpha + 1) / (N * alpha) * ((1 / rks[d1 + '_nn'] ** alpha - 1 / rks[d2 + '_nn'] ** alpha).abs()) ** (
                1 / (alpha + 1))


def rank_turbulence_divergence_sp(rks, d1, d2, N1, N2, alpha):
    """
    Compute rank turbulence divergence between date d1 and d2 for pages in rks
    :param rks: dataframe with columns d1 and d2
    :param d1: string date 1 in format YYYY-MM
    :param d2: string date 2 in format YYYY-MM
    :param N1: number of elements at d1
    :param N2: number of elements at d2
    :param alpha: hyper parameter for divergence computation
    """
    computations = rks.select(d1, d2, d1 + '_nn', d2 + '_nn', 'page')
    tmp_1 = computations.where(~col(d1).isNull()).withColumn('1/d1**alpha_N', pow(lit(1) / col(d1), lit(alpha)))
    tmp_2 = computations.where(~col(d2).isNull()).withColumn('1/d2**alpha_N', pow(lit(1) / col(d2), lit(alpha)))
    tmp_1 = tmp_1.withColumn('diff_N_1',
                             pow(abs(col('1/d1**alpha_N') - lit((1 / (N1 + 0.5 * N2)) ** alpha)), lit(1 / (alpha + 1))))
    tmp_2 = tmp_2.withColumn('diff_N_2',
                             pow(abs(col('1/d2**alpha_N') - lit((1 / (N2 + 0.5 * N1)) ** alpha)), lit(1 / (alpha + 1))))
    N = (alpha + 1) / alpha * (tmp_1.select(sum('diff_N_1').alias('dn1')).collect()[0][0] +
                               tmp_2.select(sum('diff_N_2').alias('dn2')).collect()[0][0])

    computations = computations.withColumn('1/d1**alpha', pow(lit(1) / col(d1 + '_nn'), lit(alpha)))
    computations = computations.withColumn('1/d2**alpha', pow(lit(1) / col(d2 + '_nn'), lit(alpha)))
    computations = computations.withColumn(f'div_{d2}',
                                           pow(abs(col('1/d1**alpha') - col('1/d2**alpha')), lit(1 / (alpha + 1))) * (
                                                       alpha + 1) / (alpha * N))

    return computations.withColumn('date', lit(d2)).select('page', col(f'div_{d2}').alias('div'), 'date') #, col(f'{d1}_nn').alias(f'rank_{d1}'),
                                                    # col(f'{d2}_nn').alias(f'rank_{d2}'))


def augment_div(df, rg_rk, dates, df_ranks):
    """
    Augment divergence dataframe with other statistics
    :param df: divergence dataframe
    :param rg_rk: ranks bucket number (ie. 0, 1, etc...)
    :param dates: selected dates for analysis in format YYYY-MM
    :param df_ranks: ranks dataframe
    """
    div_dates = [f"div_{d}" for d in dates]

    df_ranks_filt = df_ranks.where(df_ranks.rank_range == rg_rk).groupby('page').pivot('date').sum('rank').toPandas()

    med_divs = df.set_index('page')[div_dates].mean(axis=1)
    med_divs.name = 'avg_divs'
    std_divs = (df.set_index('page'))[div_dates].std(axis=1)
    std_divs.name = 'std_divs'
    med_ranks = df_ranks_filt.set_index('page')[dates].mean(axis=1)
    med_ranks.name = 'avg_ranks'
    nb_null = df_ranks_filt.set_index('page')[dates].isnull().sum(axis=1)
    nb_null.name = 'nb_null'
    std_ranks = df_ranks_filt.set_index('page')[dates].std(axis=1)
    std_ranks.name = 'std_ranks'

    return reduce(merge_index, [med_divs, std_divs, med_ranks, std_ranks, nb_null])