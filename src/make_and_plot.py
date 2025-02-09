import os
import sys
import argparse
from functools import reduce

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
import plotly.graph_objs as go
import plotly.express as px

from tqdm import tqdm

import pyspark
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark import SparkContext

sys.path.append('../')
from src.rank_turbulence_divergence import rank_turbulence_divergence_sp, RTD_0_sp, RTD_inf_sp
from src.pages_groups_extraction import extract_volume


def make_cumulative_plot(df, hinge_point, path=None):
    fig = go.Figure()
    fig.add_traces(
        go.Scatter(
            x=[hinge_point[1], hinge_point[1]],
            y=[0, 100],
            mode="lines",
            line=go.scatter.Line(color="green"),
            name=f"{np.round(hinge_point[0] * 100) / 100} % of views")
    )
    fig.add_traces(
        go.Scatter(
            y=[hinge_point[0], hinge_point[0]],
            x=[0, 100],
            mode="lines",
            line=go.scatter.Line(color="red"),
            name=f"{np.round(hinge_point[1] * 100) / 100} % of ranks")
    )
    fig.add_traces(
        go.Scatter(
            y=df['perc_views'].values,
            x=df['perc_rank'].values,
            mode="lines",
            line=go.scatter.Line(color="blue"),
            name=f"Cumulated views",
            error_y=dict(
                type='data',  # value of error bar given in data coordinates
                array=df['error_views'].values,
                visible=True)
        ))
    fig.update_layout(
        xaxis_title=dict(text='Cumulated ranks in %', font=dict(size=20)),
        yaxis_title=dict(text='Cumulated views in %', font=dict(size=20)),
        height=600,
        width=800,
        legend=dict(font=dict(size=20), title=None),
        yaxis=dict(tickfont=dict(size=20)),
        xaxis=dict(tickfont=dict(size=20)), )

    fig.show()
    if path is not None:
        fig.write_image(path)


def set_up_mapping(topics=None, grouped=True):
    """
    Create custom topic - color mapping for plots
    :param topics : list of topics
    :param grouped: boolean, if true, then group topics color by category (eg. Stem)
    :return dictionnary
    """
    if topics is None:
        with open("../src/topics_list.txt", 'r') as f:
            lines = f.read()
        topics = lines.replace('\n', '').replace("'", '').split(',')
    color_mapping = {}
    if grouped:
        viridis = mpl.colormaps['viridis'].resampled(23)  # geography
        plasma = mpl.colormaps['plasma'].resampled(22)  # culture
        spring = mpl.colormaps['spring'].resampled(7) # history
        cool = mpl.colormaps['cool'].resampled(12) # stem

        color_mapping.update({t : c for t, c in zip([t for t in topics if 'geography' in t], viridis.colors)})
        color_mapping.update({t: c for t, c in zip([t for t in topics if 'culture' in t], plasma.colors)})
        color_mapping.update({t: c for t, c in zip([t for t in topics if 'history' in t], spring(np.arange(0,spring.N)))})
        color_mapping.update({t: c for t, c in zip([t for t in topics if 'stem' in t], cool(np.arange(0,cool.N)))})

    else:
        with open("../src/colors.txt", 'r') as f:
            lines = f.read()
        colors = lines.replace('\n', '').replace("'", '').split(',')

        color_mapping.update({t : c for t, c in zip(topics, colors)})

    return color_mapping

def prepare_RTD_ranks(df, d1, d2, n=int(1e8)):

    """
    Prepare ranking heatmap plot
    :param df: Pyspark DataFrame
        contains all data for dates d1 and d2
    :param d1: str
        date 1
    :param d2: str
        date 2
    :param n: int
        number of pages to consider in the plot, top n pages
    :return df_comparison : pyspark Dataframe with ranking comparisons between the two dates d1 and d2
            N1, N2 : number of pages considered in resp. d1, d2
            N : number of pages in common between d1 and d2
    """

    df_piv = df.sort(asc('fractional_rank')) \
        .limit(n) \
        .groupBy('page_id').pivot('date').sum('fractional_rank')

    df_comparison = df_piv.where(~col(d2).isNull() | ~col(d1).isNull()).cache()

    N1 = df_comparison.where(~col(d1).isNull()).count()
    N2 = df_comparison.where(~col(d2).isNull()).count()
    N = df_comparison.where(~col(d2).isNull() & ~col(d1).isNull()).count()

    last_rk1 = N1 + 0.5 * (N2 - N)
    last_rk2 = N2 + 0.5 * (N1 - N)
    df_comparison = df_comparison.withColumn(d1 + '_nn', col(d1)).withColumn(d2 + '_nn', col(d2))
    df_comparison = df_comparison.fillna({d1 + '_nn': last_rk1, d2 + '_nn': last_rk2})

    return df_comparison.select(col(d1).alias('prev_rank_1'),
                                col(d2).alias('prev_rank_2'),
                                col(d1 + '_nn').alias('rank_1'),
                                col(d2 + '_nn').alias('rank_2'),
                                'page_id'),\
           N1, N2, N


def prepare_heat_map(df, prev_date, next_date, n, debug=False):

    """
    Prepare Rank Turbulence divergence heatmap plot
    :param df: pyspark DataFrame
            dataframe with pageviews, viewcounts, ranks and dates
    :param prev_date: str
    :param next_date: str
    :param n: int
            top n pages to consider for the plot
    """
    if debug : print("# Prepare heatmap")
    if debug : print("Prepare ranks")
    df_ranked, N1, N2, N = prepare_RTD_ranks(df.where(df.date.isin([prev_date, next_date])),
                                             prev_date,
                                             next_date,
                                             n=n)

    if debug : print("Log ranks and find max")
    df_ranked = df_ranked.withColumn('log_rank_1', round(log10(col('rank_1')) * 100) / 100)\
        .withColumn('log_rank_2',round(log10(col('rank_2')) * 100) / 100).cache()  # Keep first digits after coma

    if debug : print("Match page with titles")
    df_ranked = df_ranked.join(dfs.select('page', 'page_id').distinct(), 'page_id').dropDuplicates(
        ['page_id', 'rank_1', 'rank_2']).cache()  # There are duplicates

    if debug : print("Build up plot legends and counts")
    w1 = Window.partitionBy(['log_rank_1', 'log_rank_2']).orderBy(asc('rank_1'))
    w2 = Window.partitionBy(['log_rank_1', 'log_rank_2']).orderBy(asc('rank_2'))

    df_agg = df_ranked.groupBy('log_rank_1', 'log_rank_2').agg(count('*').alias('nb_pages'))
    df_select_first = df_ranked.withColumn('page_1', first('page').over(w1)) \
        .withColumn('r_1_1', first('rank_1').over(w1)) \
        .withColumn('r_1_2', first('rank_2').over(w1))
    df_select_first = df_select_first.withColumn('page_2', first('page').over(w2)) \
        .withColumn('r_2_1', first('rank_1').over(w2)) \
        .withColumn('r_2_2', first('rank_2').over(w2)) \
        .dropDuplicates(['log_rank_1', 'log_rank_2']) \
        .drop('prev_rank_1', 'rank_1', 'prev_rank_2', 'rank_2', 'page_id', 'page')
    df_agg = df_agg.join(df_select_first, on=['log_rank_1', 'log_rank_2']).cache()

    if debug : print("Heatmap plot")
    df_plot = df_agg.drop('page').toPandas()
    df_plot['label'] = df_plot.apply(
        lambda r: f"{r['page_1']} : {r['r_1_1']} - {r['r_1_2']} <br>{r['page_2']} : {r['r_2_1']} - {r['r_2_2']}<br>",
        axis=1)
    df_plot['date'] = f"{prev_date} / {next_date}"
    df_plot['set_size'] = n

    return df_ranked, df_plot, N1, N2, N


def prepare_divergence_plot(df, alpha, prev_date, next_date, n, N1, N2, lim=1000, nb_top_pages=40, debug=False, make_plot=True):

    if debug : print("Compute divergence")
    if alpha == 0:
        df_divs = RTD_0_sp(
            df.select('page_id', col('rank_1').alias(f'{prev_date}_nn'), col('rank_2').alias(f'{next_date}_nn'), col('prev_rank_1').alias(prev_date), col('prev_rank_2').alias(next_date), 'page'),
            prev_date,
            next_date,
            N1, N2).cache()
    elif np.isinf(alpha):
        df_divs = RTD_inf_sp(
            df.select('page_id', col('rank_1').alias(f'{prev_date}_nn'), col('rank_2').alias(f'{next_date}_nn'), col('prev_rank_1').alias(prev_date), col('prev_rank_2').alias(next_date), 'page'),
            prev_date,
            next_date).cache()
    else:
        df_divs = rank_turbulence_divergence_sp(
            df.select('page_id', col('rank_1').alias(f'{prev_date}_nn'), col('rank_2').alias(f'{next_date}_nn'), col('prev_rank_1').alias(prev_date), col('prev_rank_2').alias(next_date), 'page'),
            prev_date,
            next_date,
            N1, N2,
            alpha=alpha).cache()

    if make_plot:
        df_div_pd = df_divs.sort(desc('div')).limit(lim).toPandas()

        if debug : print("Find exclusive types ranks")
        max_rk_1 = df_divs.select(max('rank_1').alias('m_1')).collect()[0]['m_1']
        max_rk_2 = df_divs.select(max('rank_2').alias('m_2')).collect()[0]['m_2']

        if debug : print("Find plotting settings")
        # Save settings
        df_div_pd['alpha'] = alpha
        df_div_pd['set_size'] = n
        # For labelling
        df_div_pd['ranks'] = df_div_pd.apply(lambda r: f"{int(r['rank_1'])} <> {int(r['rank_2'])}", axis=1)
        # Note the exclusive types with an asterix
        df_div_pd['page'] = df_div_pd.apply(lambda r: r.page + str('*') if ((r['rank_1'] == max_rk_1) | (r['rank_2'] == max_rk_2)) else r.page, axis=1)

        # Take the top divergence for both dates
        df_div_pd['div_sign'] = df_div_pd.apply(lambda r: (2 * int(r['rank_2'] < r['rank_1']) - 1) * r[f'div'], axis=1)
        df_plot_head = df_div_pd.sort_values(by=f'div_sign', ascending=False)[
            ['div', 'div_sign', 'page', 'ranks', 'alpha', 'set_size']] \
            .head(nb_top_pages // 2)
        df_plot_tail = df_div_pd.sort_values(by=f'div_sign', ascending=False)[
            ['div', 'div_sign', 'page', 'ranks', 'alpha', 'set_size']] \
            .tail(nb_top_pages // 2)
        df_plot = pd.concat([df_plot_head, df_plot_tail])

        # labels
        df_plot['month'] = [prev_date if s < 0 else next_date for s in df_plot['div_sign'].values]
        df_plot['date'] = f"{prev_date} / {next_date}"

        return df_plot, df_divs

    else:
        return df_divs

def prepare_stats(df_rank, df_div, dfs, prev_date, next_date, alpha, n):

    # Exclusive types percentage
    nb_exclusive_1 = df_rank.where(col('prev_rank_2').isNull()).count()
    nb_exclusive_2 = df_rank.where(col('prev_rank_1').isNull()).count()

    # Counts percentage
    nb_counts_1 = dfs.where(f'date = "{prev_date}"').select(sum('tot_count_views').alias('sum')).collect()[0]['sum']
    nb_counts_2 = dfs.where(f'date = "{next_date}"').select(sum('tot_count_views').alias('sum')).collect()[0]['sum']

    # Items percentage
    nb_pages_1 = df_rank.where(col('prev_rank_1').isNotNull()).count()
    nb_pages_2 = df_rank.where(col('prev_rank_2').isNotNull()).count()

    # Div contribution
    div_1 = df_div.where(col('rank_1') > col('rank_2')).select(sum('div').alias('sum')).collect()[0]['sum']
    div_2 = df_div.where(col('rank_1') <= col('rank_2')).select(sum('div').alias('sum')).collect()[0]['sum']

    stats = pd.DataFrame({'Category': ['exclusive type', 'exclusive type', 'views', 'views', 'number of pages',
                                       'number of pages', 'divergence', 'divergence'],
                          'counts': [nb_exclusive_1, nb_exclusive_2, nb_counts_1, nb_counts_2, nb_pages_1, nb_pages_2,
                                     div_1, div_2],
                          'Month': [prev_date, next_date, prev_date, next_date, prev_date, next_date, prev_date, next_date]})
    stats_agg = stats.groupby('Category').agg({'counts': 'sum'}).rename(columns={'counts': 'sum'}).reset_index()
    stats = stats.merge(stats_agg, on='Category')
    stats['Percentage of total contribution'] = stats['counts'] / stats['sum'] * 100

    stats['set_size'] = n
    stats['alpha'] = alpha
    stats['date'] = f"{prev_date} / {next_date}"

    return stats


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',
                        type=str,
                        choices=['date', 'alpha', 'rtd'],
                        default='date')
    parser.add_argument('--alpha',
                        type=float,
                        default=0.3)
    parser.add_argument('--memory',
                        default=70,
                        type=int,
                        choices=[30, 50, 70, 100, 120])
    parser.add_argument('--cores',
                        type=int,
                        choices=[1,2,3,4,5,6,7,8,9,10],
                        default=5)
    args = parser.parse_args()

    # Spark
    conf = pyspark.SparkConf().setMaster(f"local[{args.cores}]").setAll([
        ('spark.driver.memory', f'{args.memory}G'),
        ('spark.executor.memory', f'{args.memory}G'),
        ('spark.driver.maxResultSize', '0'),
        ('spark.executor.cores', f'{args.cores}'),
        ('spark.local.dir', '/scratch/descourt/spark')
    ])
    # create the session
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    # create the context
    sc = spark.sparkContext
    sc.setLogLevel('ERROR')

    dfs = spark.read.parquet("/scratch/descourt/processed_data/en/pageviews_en_2015-2023.parquet")\
        .withColumn('project', lit('en'))
    plot_dir = "/scratch/descourt/plots/thesis"
    os.makedirs(plot_dir, exist_ok=True)

    # Matching page ids across time based on last title to avoid pages popping up
    dfs_change_all = spark.read.parquet("/scratch/descourt/processed_data/en/pageviews_en_articles_ev_2023-03.parquet")

    # Plot dataframes
    df_plot_heatmap = []
    df_plot_divs = []
    df_stats = []

    if args.mode == 'rtd':

        df = extract_volume(dfs, high=True).cache()
        df = df.join(dfs_change_all.select('last_page_id', 'page_ids', 'last_name'),
                       dfs_change_all.page_ids == df.page_id) \
            .select('date', col('last_page_id').alias('page_id'), col('last_name').alias('page'), 'fractional_rank').cache()

        dates = ['2015-07', '2015-08', '2015-09', '2015-10', '2015-11', '2015-12']\
                + [f'{y}-{m}' for y in ['2016', '2017', '2018', '2019', '2020', '2021', '2022'] for m in
                               [f'0{i}' if i < 10 else i for i in range(1, 13, 1)]]\
                + ['2023-01', '2023-02', '2023-03']
        prev_d = dates[:-1]
        next_d = dates[1:]

        for p, n in tqdm(zip(prev_d, next_d)):
            df_ranked, N1, N2, N = prepare_RTD_ranks(df.where(df.date.isin([p, n])),
                                                   p,
                                                   n,
                                                   n=10**7)
            df_ranked = df_ranked.join(df.select('page', 'page_id').distinct(), 'page_id')
            df_divs = prepare_divergence_plot(df_ranked, args.alpha, p, n, int(10**7), N1, N2, make_plot=False)

            df_plot_divs.append(df_divs)

        reduce(DataFrame.unionAll, df_plot_divs).write.parquet(os.path.join(plot_dir, 'RTD_all.parquet'))

    if args.mode == 'date':

        dates = ['2019-12'] + [f'{y}-{m}' for y in ['2020'] for m in
                               [f'0{i}' if i < 10 else i for i in range(1, 13, 1)]]
        prev_d = dates[:-1]
        next_d = dates[1:]

        dfs = dfs.join(dfs_change_all.select('last_page_id', 'page_ids', 'last_name'),
                       dfs_change_all.page_ids == dfs.page_id) \
            .select('date', col('last_page_id').alias('page_id'), col('last_name').alias('page'), 'fractional_rank').cache()

        for p, n in tqdm(zip(prev_d, next_d)):

            df_ranked, df_heatmap, N1, N2, N = prepare_heat_map(dfs, p, n, int(10**8))
            df_div_pd, df_divs = prepare_divergence_plot(df_ranked, args.alpha, p, n, int(10**8), N1, N2)
            stats = prepare_stats(df_ranked, df_divs, dfs, p, n, args.alpha, int(10**8))

            df_plot_heatmap.append(df_heatmap)
            df_plot_divs.append(df_div_pd)
            df_stats.append(stats)

        pd.concat(df_plot_heatmap).to_csv(os.path.join(plot_dir, 'heatmap_dates.csv.gzip'), compression='gzip')
        pd.concat(df_plot_divs).to_csv(os.path.join(plot_dir, 'divs_dates.csv.gzip'), compression='gzip')
        pd.concat(df_stats).to_csv(os.path.join(plot_dir, 'stats_dates.csv'))


    elif args.mode == 'alpha':

        # Alphas
        alphas = [0, 0.3, np.inf] # TODO put back to full range for animations !
        # Dates to compare
        p = '2021-11'
        n = '2021-12'

        # Match page ids to avoid pages popping up
        dfs = dfs.join(dfs_change_all.select('last_page_id', 'page_ids', 'last_name'),
                       dfs_change_all.page_ids == dfs.page_id) \
            .select('date', col('last_page_id').alias('page_id'), col('last_name').alias('page'), 'fractional_rank').cache()
        df_ranked, N1, N2, N = prepare_RTD_ranks(dfs.where(dfs.date.isin([p, n])),
                                                 p,
                                                 n,
                                                 n=10 ** 8)
        # Retrieve page titles
        df_ranked = df_ranked.join(dfs.select('page', 'page_id').distinct(), 'page_id').cache()

        df_divs_sp = []

        for alpha in tqdm(alphas):
            df_div_pd, df_divs = prepare_divergence_plot(df_ranked, alpha=alpha, prev_date=p,
                                                         next_date=n, n=int(10**8), N1=N1, N2=N2, make_plot=True)

            df_plot_divs.append(df_div_pd)
            df_divs_sp.append(df_divs.withColumn('alpha', lit(alpha)))

        pd.concat(df_plot_divs).to_csv(os.path.join(plot_dir, 'divs_alphas.csv.gzip'), compression='gzip')
        reduce(DataFrame.unionAll, df_divs_sp).write.parquet(os.path.join(plot_dir, 'RTD_alphas_dec21.parquet'))

    print('Done')










