import os
from pyspark.sql.functions import *


os.environ["JAVA_HOME"] = "/lib/jvm/java-11-openjdk-amd64"


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

