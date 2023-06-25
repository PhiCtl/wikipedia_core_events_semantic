from functools import reduce

import numpy as np
from tqdm import tqdm
import os
import pyspark
from pyspark.sql import *
from pyspark.sql.functions import *

import sys
sys.path.append('../')
from src.ranking_helpers import merge_index
from src.pages_groups_extraction import extract_volume

os.environ["JAVA_HOME"] = "/lib/jvm/java-11-openjdk-amd64"


def rank_turbulence_divergence_sp(rks, d1, d2, alpha):
    """
    Compute rank turbulence divergence between date d1 and d2 for pages in rks
    :param rks: dataframe with columns d1 and d2
    :param d1: string date 1 in format YYYY-MM
    :param d2: string date 2 in format YYYY-MM
    :param N1: number of elements at d1
    :param N2: number of elements at d2
    :param alpha: hyper parameter for divergence computation
    """
    computations = rks.select(d1, d2, d1 + '_nn', d2 + '_nn', 'page_id' ,'page',
                              'n1', 'n2', 'n', 'topic')

    tmp_1 = computations.where(~col(d1).isNull())\
                        .withColumn('1/d1**alpha_N', pow(1 / col(d1), alpha))\
                        .withColumn('diff_N_1',
                             pow(
                                 abs(col('1/d1**alpha_N') - (1 / (col('n1') + 0.5 * col('n2'))) ** alpha),
                             lit(1 / (alpha + 1))))

    tmp_2 = computations.where(~col(d2).isNull())\
                        .withColumn('1/d2**alpha_N', pow(1 / col(d2), alpha))\
                        .withColumn('diff_N_2',
                            pow(
                                abs(col('1/d2**alpha_N') - (1 / (col('n2') + 0.5 * col('n1'))) ** alpha),
                                lit(1 / (alpha + 1))))

    Ns = tmp_1.groupBy('topic').agg(sum('diff_N_1').alias('dn1')) \
        .join(tmp_2.groupBy('topic').agg(sum('diff_N_2').alias('dn2')), 'topic') \
        .select('topic', ((col('dn1') + col('dn2')) * ((alpha + 1) / alpha)).alias('N'))

    computations = computations.withColumn('1/d1**alpha', pow(1 / col(d1 + '_nn'), alpha))
    computations = computations.withColumn('1/d2**alpha', pow(1 / col(d2 + '_nn'), alpha)).drop('n', 'n1', 'n2')

    computations = computations.join(Ns, on='topic')\
                               .withColumn(f'div',
                                           pow(abs(col('1/d1**alpha') - col('1/d2**alpha')), 1 / (alpha + 1))
                                           * (alpha + 1) / (alpha * col('N')))

    return computations.withColumn('date', lit(d2)).select('div', 'date', 'page_id', 'page','topic',
                                                           col(f'{d1}_nn').alias('rank_1'),
                                                           col(f'{d2}_nn').alias('rank_2'),
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
    computations = rks.select(d1, d2, d1 + '_nn', d2 + '_nn', 'page_id', 'page')
    tmp_1 = computations.where(~col(d1).isNull()).withColumn('abs_ln_r1_Ns', abs(log(col(d1) / (N1 + 1 / 2 * N2))))
    tmp_2 = computations.where(~col(d2).isNull()).withColumn('abs_ln_r2_Ns', abs(log(col(d2) / (N2 + 1 / 2 * N1))))

    N = tmp_1.select(sum('abs_ln_r1_Ns').alias('dn1')).collect()[0][0] + \
        tmp_2.select(sum('abs_ln_r2_Ns').alias('dn2')).collect()[0][0]

    computations = computations.withColumn('abs_ln_r1_r2', abs(log(col(d1 + '_nn') / col(d2 + '_nn'))))
    computations = computations.withColumn(f'div_{d2}', col('abs_ln_r1_r2') / N)

    return computations.withColumn('date', lit(d2)).select(col(f'div_{d2}').alias('div'), 'date', 'page_id', 'page',
                                                           col(f'{d1}_nn').alias('rank_1'),
                                                           col(f'{d2}_nn').alias('rank_2'),
                                                           col(d1).alias('prev_rank_1'), col(d2).alias('prev_rank_2'))


def RTD_inf_sp(rks, d1, d2):
    """
    Compute rank turbulence divergence for alpha = inf between date d1 and d2 for pages in rks
    :param rks: dataframe with columns d1 and d2
    :param d1: string date 1 in format YYYY-MM
    :param d2: string date 2 in format YYYY-MM
    """
    computations = rks.select(d1, d2, d1 + '_nn', d2 + '_nn', 'page_id' ,'page')
    tmp_1 = computations.where(~col(d1).isNull()).withColumn('1_r1', 1 / col(d1))
    tmp_2 = computations.where(~col(d2).isNull()).withColumn('1_r2', 1 / col(d2))

    N = tmp_1.select(sum('1_r1').alias('dn1')).collect()[0][0] + tmp_2.select(sum('1_r2').alias('dn2')).collect()[0][0]

    computations = computations.withColumn('max_r1_r2', when(col(d1 + '_nn') == col(d2 + '_nn'), 0).otherwise(
        greatest(1 / col(d1 + '_nn'), 1 / col(d2 + '_nn'))))
    computations = computations.withColumn(f'div_{d2}', col('max_r1_r2') / N)

    return computations.withColumn('date', lit(d2)).select(col(f'div_{d2}').alias('div'), 'date', 'page_id', 'page',
                                                           col(f'{d1}_nn').alias('rank_1'),
                                                           col(f'{d2}_nn').alias('rank_2'),
                                                           col(d1).alias('prev_rank_1'), col(d2).alias('prev_rank_2'))



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

    save_path = "/scratch/descourt/processed_data/en"
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
