from functools import reduce

from tqdm import tqdm

import os

os.environ["JAVA_HOME"] = "/lib/jvm/java-11-openjdk-amd64"

import pyspark
from pyspark.sql import *
from pyspark.sql.functions import *

from ranking_helpers import compute_fractional_ranking, rank_turbulence_divergence_sp
from data_aggregation import *
from volumes_extraction import extract_volumes, find_hinge_point

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

from functools import reduce


if __name__ == '__main__':

    save_path = "/scratch/descourt/processed_data_050923"
    os.makedirs(save_path, exist_ok=True)

    # path
    path = '/scratch/descourt/interm_results/rank_div_fract'
    os.makedirs(path, exist_ok=True)

    try:

        # Data - all
        # print('Compute frac ranks')
        # dfs_prev = spark.read.parquet(os.path.join(save_path, save_file))
        # # Compute fractional_ranking
        # dfs_prev = compute_fractional_ranking(dfs_prev)
        # dfs_prev.write.parquet(os.path.join(save_path, "pageviews_agg_en_2015-2023_frac.parquet"))
        dfs = spark.read.parquet(os.path.join(save_path, "pageviews_en_2015-2023_frac.parquet")).select(col('fractional_rank').alias('rank'), 'page', 'page_id', 'date', 'tot_count_views')

        # Extract high volume core
        df_vols = extract_volumes(dfs)
        # Find hinge point
        df_hinge = find_hinge_point(df_vols).cache()
        # Take all pages which are below hinge point for views
        df_high_volume = df_vols.join(df_hinge,'date').where(col('perc_views') <= col('hinge_perc'))

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
