import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pyspark.sql import *
from pyspark.sql.functions import *



def find_hinge_point_np(y):
    # Assumes curve is not noisy

    slope = 100 / len(y)
    h_idx = np.where(np.gradient(y) <= slope)[0][0]
    return h_idx / len(y) * 100, y[h_idx]

def plot_volume_curve(df, label):
    plt.figure()
    df.plot(x='perc_rank', y='perc_views', label=label)
    idx_h, h = find_hinge_point_np(df['perc_views'].values)
    plt.axhline(h, c='g')
    plt.axvline(idx_h, c='g', label=f'{int(h)}%-{idx_h}% of total rank')
    plt.title(f'Cumulative views distribution of Wikipedia pages for {label}')
    plt.xlabel('Percentage of last rank')
    plt.ylabel('Cumulative percentage of pageviews monthly volume')
    plt.legend()



def compute_gradient(df, x):
    # We do not compute boundaries derivative for each month
    # We use central finite difference, assuming regular (1) spacing

    w = Window.partitionBy('date').orderBy('rank')
    df_h = df.withColumn('p', lag(x).over(w)) \
        .withColumn('n', lead(x).over(w)) \
        .where(~col('p').isNull() & ~col('n').isNull())
    df_h = df_h.withColumn(f'd_{x}', (col('n') - col('p')) / 2)

    return df_h.drop('p', 'n', 'h_p', 'h_n')


def find_hinge_point(df, eps=1):
    df_grad = compute_gradient(df, x='perc_views')
    slopes = df.groupBy('date').agg(count('*').alias('slope')).select(col('date').alias('d'),
                                                                      (100 / col('slope') * eps).alias('slope'))
    df_hinge = df_grad.join(slopes, slopes.d == df_grad.date) \
        .drop('d') \
        .sort(asc('rank')) \
        .filter(col('d_perc_views') <= col('slope')) \
        .groupBy('date').agg(first('perc_views').alias('hinge_perc'), first('perc_rank').alias('hinge_rank'))
    return df_hinge

def compute_volumes(df, partition='date', sampling=1):

    window = Window.partitionBy(partition).orderBy('rank')
    # Sample for robustness tests
    df = df.where((col('rank') % sampling == 0) | (col('rank') == 1))
    df_cutoff = df.withColumn('cum_views', sum('tot_count_views').over(window))
    df_sum = df.groupBy(partition).agg(sum('tot_count_views').alias('tot_counts'), count('*').alias('tot_nb_pages'))
    df_cutoff = df_cutoff.join(df_sum, partition) \
        .withColumn('perc_views', col('cum_views') / col('tot_counts') * 100) \
        .withColumn('perc_rank', col('rank') / col('tot_nb_pages') * 100).drop('tot_nb_pages', 'tot_counts')

    return df_cutoff

def extract_volume(df, high=True, sampling=1, eps=1):
    df_cutoff = compute_volumes(df, sampling=sampling)
    # Find hinge point
    df_hinge = find_hinge_point(df_cutoff, eps=eps).cache()
    # Take all pages which are below hinge point for views
    if high :
        df_volume = df_cutoff.join(df_hinge, 'date').where(col('perc_views') <= col('hinge_perc'))
    else:
        # Low
        df_volume = df_cutoff.join(df_hinge, 'date').where(col('perc_views') > col('hinge_perc'))
    return df_volume


def extract_common_pages(df, time_period=None, aggregate=False, nb_dates=None):
    if time_period is not None:
        df = df.where(df.date.isin(time_period))

    # Count number of occurences for each page in the considered time period
    df = df.join(df.groupBy('page_id').agg(count('*').alias('nb_occurrences')), 'page_id').cache()
    if nb_dates is None:
        nb_dates = df.select('date').distinct().count()

    # Extract pages which are present during the entire time period and aggregate several metrics
    df_stable = df.where(df.nb_occurrences == nb_dates).join(df.select('date', 'page_id'), on=['date', 'page_id'])
    if aggregate:
        df_stable = df_stable.sort(asc('date')).groupBy('page_id').agg(
            # collect_list('div').alias('div_sorted_list'),\
            avg('div').alias('avg_div'), \
            stddev('div').alias('std_div'), \
            avg('tot_count_views').alias('avg_pageviews'), \
            stddev('tot_count_views').alias('std_pageviews'), \
            avg('rank').alias('avg_rank'), \
            stddev('rank').alias('std_rank'))
    return df_stable






