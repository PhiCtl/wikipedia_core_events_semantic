import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pyspark.sql import *
from pyspark.sql.functions import *


def compute_gradient(df, x):

    """
    Compute spatial derivative d_x / d_h, with h spatial step of size 1
    :param df : DataFrame
    :param x : str, name of the column from which we want to take the 1st order derivative

    Note :
    - We do not compute boundaries derivative for each month
    - We use central finite difference, assuming regular (1) spacing
    """

    w = Window.partitionBy('project', 'date').orderBy('rank')
    df_h = df.withColumn('p', lag(x).over(w)) \
        .withColumn('n', lead(x).over(w)) \
        .where(~col('p').isNull() & ~col('n').isNull())
    df_h = df_h.withColumn(f'd_{x}', (col('n') - col('p')) / 2)

    return df_h.drop('p', 'n', 'h_p', 'h_n')


def find_hinge_point(df, eps=1):
    """
    Find hinge point :
    The hinge can be defined as the point on the curve where we start adding “more ranks than views”
    in terms of contribution.
    :param df : DataFrame
    :param eps : inclination of the slope tangent to the hinge point. Default = 1
    """
    df_grad = compute_gradient(df, x='perc_views')
    slopes = df.groupBy('project', 'date').agg(count('*').alias('slope')).select('date', 'project',
                                                                      (100*eps / col('slope')).alias('slope'))
    df_hinge = df_grad.join(slopes, on=['project', 'date']) \
        .sort(asc('rank')) \
        .filter(col('d_perc_views') <= col('slope')) \
        .groupBy('project', 'date').agg(first('perc_views').alias('hinge_perc'), first('perc_rank').alias('hinge_rank'))
    return df_hinge

def compute_volumes(df, sampling=1):
    """
    Compute the ordered cumulated views and ranks for core / tail extraction using the hinge method
    :param df: DataFrame (pyspark)
    :param sampling: int, sampling step of the full dataframe
    """

    window = Window.partitionBy('project', 'date').orderBy('rank')
    # Sample for robustness tests
    df = df.where((col('rank') % sampling == 0) | (col('rank') == 1))
    df_cutoff = df.withColumn('cum_views', sum('tot_count_views').over(window))
    df_sum = df.groupBy('project', 'date').agg(sum('tot_count_views').alias('tot_counts'), (count('*') * sampling).alias('tot_nb_pages'))
    df_cutoff = df_cutoff.join(df_sum, on=['date', 'project']) \
        .withColumn('perc_views', col('cum_views') / col('tot_counts') * 100) \
        .withColumn('perc_rank', col('rank') / col('tot_nb_pages') * 100).drop('tot_nb_pages', 'tot_counts')

    return df_cutoff

def extract_volume(df, high=True, sampling=1, eps=1):
    """
    Extract core or tail using the hinge method
    :param df: dataframe
    :param high: boolean, whether to extract high or low volume
    :param sampling: int, sampling step of the full dataframe
    :param eps: inclination of the slope at the hinge point
    """
    df_cutoff = compute_volumes(df, sampling=sampling)
    # Find hinge point
    df_hinge = find_hinge_point(df_cutoff, eps=eps)
    # Take all pages which are below hinge point for views
    if high :
        df_volume = df_cutoff.join(df_hinge, on=['date', 'project']).where(col('perc_views') <= col('hinge_perc'))
    else:
        # Low
        df_volume = df_cutoff.join(df_hinge, on=['date', 'project']).where(col('perc_views') > col('hinge_perc'))
    return df_volume







