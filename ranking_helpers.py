from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

from pyspark.sql import *
from pyspark.sql.functions import *
import pyspark

from pyspark.sql.window import Window
from pyspark.ml.feature import QuantileDiscretizer
from pyspark.ml.feature import Bucketizer


def compute_ranks_bins(df, lim=100000, slicing=5000, subset='date'):
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
    bucketizer = Bucketizer(splits=[i for i in range(0, lim, slicing)] + [lim], inputCol="rank", outputCol="rank_range")
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


def slice_comparison(df, dates, mapping, title="between consecutive periods of 2 months"):
    """
    Plot overlap for several dates over rank ranges
    :param title: title completion
    :param df:
    :param dates: dates (or date ranges if dates are binned) to select
    :param mapping: to convert dates or date ranges into readable legends
    :return:
    """
    df_plot = df.filter(df.date_range.isin(dates)).toPandas()
    df_plot['date_range'].replace(mapping, inplace=True)
    df_plot = df_plot.pivot(index='rank_range', columns='date_range', values='overlap')
    plt.figure()
    df_plot.plot(colormap='Paired')
    plt.xlabel('Rank range')
    plt.ylabel('Overlap between two series in %')
    plt.xticks()
    plt.yticks()
    plt.title(f"Normalized percentage overlap {title} per rank range")
    plt.legend(bbox_to_anchor=(1.3, 1), title="date range")
    plt.show()