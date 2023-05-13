import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pyspark.sql.functions import pow
from pyspark.sql import *
from pyspark.sql.functions import *
import pyspark


def find_inflection_point(y):
    # Assumes curve is not noisy
    # Gives us 0

    dy = np.gradient(y)
    m_idx = np.argmax(dy)
    return m_idx, y[m_idx]


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


def find_hinge_point(df):
    df_grad = compute_gradient(df, x='perc_views')
    slopes = df.groupBy('date').agg(count('*').alias('slope')).select(col('date').alias('d'),
                                                                      (100 / col('slope')).alias('slope'))
    df_hinge = df_grad.join(slopes, slopes.d == df_grad.date) \
        .drop('d') \
        .sort(asc('rank')) \
        .filter(col('d_perc_views') <= col('slope')) \
        .groupBy('date').agg(first('perc_views').alias('hinge_perc'), first('perc_rank').alias('hinge_rank'))
    return df_hinge


def find_inflection_point(df):
    df_grad = compute_gradient(df, x='perc_views')
    df_lapl = compute_gradient(df_grad, x='d_perc_views')
    df_inflection = df_lapl.sort(asc('rank')) \
        .filter(col('d_d_perc_views') <= 0) \
        .groupBy('date') \
        .agg(first('perc_views').alias('inf_perc'), first('rank').alias('infl_rank'))
    return df_inflection

def compute_volumes(df):
    window = Window.partitionBy('date').orderBy('rank')

    df_cutoff = df.withColumn('cum_views', sum('tot_count_views').over(window))
    df_sum = df.groupBy('date').agg(sum('tot_count_views').alias('tot_count_month'), count('*').alias('nb_pages'))
    df_cutoff = df_cutoff.join(df_sum, 'date') \
        .withColumn('perc_views', col('cum_views') / col('tot_count_month') * 100) \
        .withColumn('perc_rank', col('rank') / col('nb_pages') * 100).drop('nb_pages')

    return df_cutoff

def extract_volume(df, high=True):
    df_cutoff = compute_volumes(df)
    # Find hinge point
    df_hinge = find_hinge_point(df_cutoff).cache()
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






