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

def extract_volumes(df):
    window = Window.partitionBy('date').orderBy('rank')

    df_cutoff = df.withColumn('cum_views', sum('tot_count_views').over(window)) \
        .select(col('date').alias('d'), 'cum_views', 'rank', 'page')
    df_sum = df.groupBy('date').agg(sum('tot_count_views').alias('tot_count_month'), count('*').alias('nb_pages'))
    df_cutoff = df_cutoff.join(df_sum, df_sum.date == df_cutoff.d) \
        .withColumn('perc_views', col('cum_views') / col('tot_count_month') * 100) \
        .withColumn('perc_rank', col('rank') / col('nb_pages') * 100) \
        .drop('d')

    return df_cutoff



