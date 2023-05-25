from functools import reduce

import numpy as np
from tqdm import tqdm
import os
import pyspark
from pyspark.sql import *
from pyspark.sql.functions import *

from ranking_helpers import merge_index
from pages_groups_extraction import extract_volume

os.environ["JAVA_HOME"] = "/lib/jvm/java-11-openjdk-amd64"



def rank_diversity(df, rank_type='rank'):
    """
    Compute rank diversity given rank_type ranking
    rank_type can be any from 'rank', 'fract_rank', 'rank_range'
    """
    df_T = df.groupBy(rank_type).agg(count('*').alias('tot_possible'))

    rank_diversity = df.join(df_T, on=rank_type) \
        .groupBy(rank_type, 'tot_possible') \
        .agg(countDistinct('page_id').alias('nb_pages')) \
        .withColumn('rank_div', col('nb_pages') / col('tot_possible'))

    return rank_diversity

def rank_change_proba(df, rank_type='rank'):
    # TODO not tried yet
    T = df.select('date').distinct().count()
    w = Window.partitionBy(rank_type).orderBy(asc('date'))
    rank_change = df.withColumn('next_page_id', lead('page_id', 1).over(w)).where(~col('next_page_id').isNull())
    rank_change = rank_change.withColumn('diff', 1 - when(col('page_id') == col('next_page_id'), 1).otherwise(0))
    rank_change_proba = rank_change.groupBy(rank_type).agg((sum('diff') / T).alias('rank_change_proba'))
    return rank_change_proba

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
    computations = rks.select(d1, d2, d1 + '_nn', d2 + '_nn', 'page_id' ,'page')
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

    return computations.withColumn('date', lit(d2)).select(col(f'div_{d2}').alias('div'), 'date', 'page_id', 'page',
                                                           col(f'{d1}_nn').alias('rank_1'), col(f'{d2}_nn').alias('rank_2'),
                                                           col(d1).alias('prev_rank_1'), col(d2).alias('prev_rank_2'))


def RTD_0_sp(rks, d1, d2, N1, N2):
    """
    Compute rank turbulence divergence for alpha = 0 between date d1 and d2 for pages in rks
    :param rks: dataframe with columns d1 and d2
    :param d1: string date 1 in format YYYY-MM
    :param d2: string date 2 in format YYYY-MM
    :param N1: number of elements at d1
    :param N2: number of elements at d2
    """
    computations = rks.select(d1, d2, d1 + '_nn', d2 + '_nn', 'page_id')
    tmp_1 = computations.where(~col(d1).isNull()).withColumn('abs_ln_r1_Ns', abs(log(col(d1) / (N1 + 1 / 2 * N2))))
    tmp_2 = computations.where(~col(d2).isNull()).withColumn('abs_ln_r2_Ns', abs(log(col(d2) / (N2 + 1 / 2 * N1))))

    N = tmp_1.select(sum('abs_ln_r1_Ns').alias('dn1')).collect()[0][0] + \
        tmp_2.select(sum('abs_ln_r2_Ns').alias('dn2')).collect()[0][0]

    computations = computations.withColumn('abs_ln_r1_r2', abs(log(col(d1 + '_nn') / col(d2 + '_nn'))))
    computations = computations.withColumn(f'div_{d2}', col('abs_ln_r1_r2') / N)

    return computations.withColumn('date', lit(d2)).select(col(f'div_{d2}').alias('div'), 'date', 'page_id')


def RTD_inf_sp(rks, d1, d2):
    """
    Compute rank turbulence divergence for alpha = inf between date d1 and d2 for pages in rks
    :param rks: dataframe with columns d1 and d2
    :param d1: string date 1 in format YYYY-MM
    :param d2: string date 2 in format YYYY-MM
    """
    computations = rks.select(d1, d2, d1 + '_nn', d2 + '_nn', 'page_id')
    tmp_1 = computations.where(~col(d1).isNull()).withColumn('1_r1', 1 / col(d1))
    tmp_2 = computations.where(~col(d2).isNull()).withColumn('1_r2', 1 / col(d2))

    N = tmp_1.select(sum('1_r1').alias('dn1')).collect()[0][0] + tmp_2.select(sum('1_r2').alias('dn2')).collect()[0][0]

    computations = computations.withColumn('max_r1_r2', when(col(d1 + '_nn') == col(d2 + '_nn'), 0).otherwise(
        greatest(1 / col(d1 + '_nn'), 1 / col(d2 + '_nn'))))
    computations = computations.withColumn(f'div_{d2}', col('max_r1_r2') / N)

    return computations.withColumn('date', lit(d2)).select(col(f'div_{d2}').alias('div'), 'date', 'page_id')


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

def RTD_alpha_0(rks, d1, d2, N1, N2):
    N = (np.log(rks.loc[~rks[d1].isnull(), d1] / (N1 + 0.5 * N2)).abs()).sum() + (
        np.log(rks.loc[~rks[d2].isnull(), d2] / (N2 + 0.5 * N1)).abs()).sum()
    rks[f'div_{d2}'] = 1 / N * np.log(rks[d1 + '_nn'] / rks[d2 + '_nn']).abs()

if __name__ == '__main__':

    conf = pyspark.SparkConf().setMaster("local[5]").setAll([
        ('spark.driver.memory', '120G'),
        ('spark.executor.memory', '120G'),
        ('spark.driver.maxResultSize', '0'),
        ('spark.executor.cores', '5'),
        ('spark.local.dir', '/scratch/descourt/spark')
    ])
    # create the session
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    # create the context
    sc = spark.sparkContext

    save_path = "/scratch/descourt/processed_data_050923"
    os.makedirs(save_path, exist_ok=True)

    # path
    path = '/scratch/descourt/interm_results/rank_div_fract'
    os.makedirs(path, exist_ok=True)

    try:

        # Data - all
        dfs = spark.read.parquet(os.path.join(save_path, "pageviews_en_2015-2023.parquet")).select(col('fractional_rank').alias('rank'), 'page', 'page_id', 'date', 'tot_count_views')

        # Extract high volume core
        df_high_volume = extract_volume(dfs, high=True)

        # consider date per date
        months = [str(m + 1) if (m + 1) / 10 >= 1 else f"0{m + 1}" for m in range(12)]
        dates = ['2015-07', '2015-08', '2015-09', '2015-10', '2015-11', '2015-12'] + \
                [f"{y}-{m}" for y in ['2016', '2017', '2018', '2019', '2020', '2021', '2022'] for m in months] + \
                ['2023-01', '2023-02', '2023-03']

        dfs_divs = []

        for i in tqdm(range(len(dates) - 1)):
            print(f"Processing {dates[i]}-{dates[i + 1]}")

            d1 = dates[i]
            d2 = dates[i + 1]
            df_ranks_piv = df_high_volume.where(df_high_volume.date.isin([d1, d2])) \
                .groupBy('page_id').pivot('date').sum('rank')

            # 2. select rows for which at least 1 non null in both columns
            df_comparison = df_ranks_piv.where(~col(d2).isNull() | ~col(d1).isNull())

            # 3. compute last rank with average formula (extracting number of non null elements in comparison date)
            N1 = df_comparison.where(~col(d1).isNull()).count()
            N2 = df_comparison.where(~col(d2).isNull()).count()
            N = df_comparison.where(~col(d2).isNull() & ~col(d1).isNull()).count()  # Intersection

            last_rk1 = N1 + 0.5 * (N2 - N)
            last_rk2 = N2 + 0.5 * (N1 - N)
            df_comparison = df_comparison.withColumn(d1 + '_nn', col(d1)).withColumn(d2 + '_nn', col(d2))
            df_comparison = df_comparison.fillna({d1 + '_nn': last_rk1, d2 + '_nn': last_rk2})

            # 4. Do calculations and store result
            df_res = rank_turbulence_divergence_sp(df_comparison, d1, d2, N1, N2, 0.0001)
            print("Writing to file...")
            df_res.write.parquet(os.path.join(path, f'rank_turb_div_{d2}.parquet'))

        print("Done !")

    except Exception as e:
        print("Error happened during RTD extraction")

    try:

        paths = [os.path.join(path, f) for f in os.listdir(path)]
        dfs = [spark.read.parquet(p) for p in paths]
        df = reduce(DataFrame.unionAll, dfs)
        df.write.parquet(os.path.join(path, 'rank_turb_div_all.parquet'))

    except Exception as e:
        print("Error occured when saving all RTD")
