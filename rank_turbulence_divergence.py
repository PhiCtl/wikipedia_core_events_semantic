from functools import reduce

from tqdm import tqdm

import os
os.environ["JAVA_HOME"] = "/lib/jvm/java-11-openjdk-amd64"

import pyspark
from pyspark.sql import *
from pyspark.sql.functions import *

from ranking_helpers import compute_ranks_bins, rank_turbulence_divergence_sp

conf = pyspark.SparkConf().setMaster("local[*]").setAll([
                                   ('spark.driver.memory','42G'),
                                   ('spark.driver.maxResultSize', '8G')
                                  ])
# create the session
spark = SparkSession.builder.config(conf=conf).getOrCreate()
# create the context
sc = spark.sparkContext

def custom_join(df1, df2):

    return df1.join(df2, "page", how='outer')

if __name__ == '__main__':

    # path
    path = '/scratch/descourt/interm_results/rank_div'
    os.makedirs(path, exist_ok=True)

    # Data - 2019-2020-2021
    dfs = spark.read.parquet("/scratch/descourt/processed_data/pageviews_agg_en.wikipedia_2019_2020_2021_2022.parquet")
    dfs = dfs.filter(~dfs.page.contains(":") & (dfs.tot_count_views >= 1)\
                     & (~dfs.date.contains('2022')))

    # Extract high volume core
    df_rank = compute_ranks_bins(dfs, slicing=1000, lim=2 * int(1e7))
    window = Window.partitionBy('date').orderBy('rank')
    df_cutoff = df_rank.withColumn('cum_views', sum('tot_count_views').over(window)) \
        .select(col('date').alias('d'), 'cum_views', 'rank', 'page')
    df_sum = df_rank.groupBy('date').agg(sum('tot_count_views').alias('tot_count_month'))
    df_cutoff = df_cutoff.join(df_sum, df_sum.date == df_cutoff.d) \
        .withColumn('perc_views', col('cum_views') / col('tot_count_month') * 100) \
        .drop('d')
    df_high_volume = df_cutoff.where(df_cutoff.perc_views <= 90.75) # Empirical

    # consider date per date
    months = [str(m + 1) if (m + 1) / 10 >= 1 else f"0{m + 1}" for m in range(12)]
    dates = [f"{y}-{m}" for y in ['2019', '2020', '2021'] for m in months]

    dfs_divs = []

    for i in tqdm(range(len(dates) - 1)):

        print(f"Processing {dates[i]}-{dates[i+1]}")

        d1 = dates[i]
        d2 = dates[i + 1]
        df_ranks_piv = df_high_volume.where(df_high_volume.date.isin([d1, d2]))\
                                     .groupBy('page').pivot('date').sum('rank')

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
        dfs_divs.append(rank_turbulence_divergence_sp(df_comparison, d1, d2, N1, N2, 0.0001))

    print("Writing to file...")
    dfs_final = reduce(custom_join, dfs_divs)
    dfs_final.write.parquet(os.path.join(path, 'rank_turb_div_all.parquet'))
    print("Done !")
