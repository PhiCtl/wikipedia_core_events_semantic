from tqdm import tqdm

import os
os.environ["JAVA_HOME"] = "/lib/jvm/java-11-openjdk-amd64"

import pyspark
from pyspark.sql import *
from pyspark.sql.functions import *

from ranking_helpers import compute_ranks_bins

conf = pyspark.SparkConf().setMaster("local[*]").setAll([
                                   ('spark.driver.memory','24g'),
                                   ('spark.driver.maxResultSize', '8G')
                                  ])
# create the session
spark = SparkSession.builder.config(conf=conf).getOrCreate()
# create the context
sc = spark.sparkContext


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

    return computations.select('page', f'div_{d2}')

if __name__ == '__main__':

    # Data
    dfs = spark.read.parquet("/scratch/descourt/processed_data/pageviews_agg_en.wikipedia_2019_2020_2021_2022.parquet")
    dfs = dfs.filter(~dfs.page.contains(":"))


    path = "/home/descourt/interm_results/low_alpha_log_all"
    os.makedirs(path, exist_ok=True)
    slicing = 1000
    lim = int(1e6)
    df_ranks = compute_ranks_bins(dfs, slicing=slicing, lim=lim, log=True)

    df_divs = None

    # 1. consider date per date
    months = months = [str(m + 1) if (m + 1) / 10 >= 1 else f"0{m + 1}" for m in range(12)]
    dates = [f"{y}-{m}" for y in ['2019', '2020'] for m in months]

    for i in tqdm(range(len(dates) - 1)):

        print(f"Processing {dates[i]}-{dates[i+1]}")

        d1 = dates[i]
        d2 = dates[i + 1]
        df_ranks_filt = df_ranks.where(df_ranks.date.isin([d1, d2])).groupBy('page').pivot('date').sum('rank')

        # 2. select rows for which at least 1 non null in both columns
        df_filt = df_ranks_filt.where(~col(d2).isNull() | ~col(d1).isNull())

        # 3. compute last rank with average formula (extracting number of non null elements in comparison date)
        N1 = df_filt.where(~col(d1).isNull()).count()
        N2 = df_filt.where(~col(d2).isNull()).count()
        N = df_filt.where(~col(d2).isNull() & ~col(d1).isNull()).count()  # Intersection

        last_rk1 = N1 + 0.5 * (N2 - N)
        last_rk2 = N2 + 0.5 * (N1 - N)
        df_filt = df_filt.withColumn(d1 + '_nn', col(d1)).withColumn(d2 + '_nn', col(d2))
        df_filt = df_filt.fillna({d1 + '_nn': last_rk1, d2 + '_nn': last_rk2})

        # 4. Do calculations and store result
        df_filt = rank_turbulence_divergence_sp(df_filt, d1, d2, N1, N2, 0.001)
        if i == 0:
            df_divs = df_filt.select('page', f'div_{d2}')
        else:
            df_tmp = df_filt.select('page', f'div_{d2}')
            df_divs = df_divs.join(df_tmp, 'page', 'outer')
            print(df_divs)

    df_divs.write.parquet(os.path.join(path, f"div_slice_all.parquet"))
