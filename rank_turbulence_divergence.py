from functools import reduce

from tqdm import tqdm

import os
os.environ["JAVA_HOME"] = "/lib/jvm/java-11-openjdk-amd64"

import pyspark
from pyspark.sql import *
from pyspark.sql.functions import *

from ranking_helpers import compute_ranks_bins, rank_turbulence_divergence_sp
from data_aggregation import*

conf = pyspark.SparkConf().setMaster("local[5]").setAll([
                                   ('spark.driver.memory','120G'),
                                   ('spark.executor.memory', '120G'),
                                   ('spark.driver.maxResultSize', '0'),
                                    ('spark.executor.cores', '5'),
                                    ('spark.local.dir', '/scratch/descourt')
                                  ])
# create the session
spark = SparkSession.builder.config(conf=conf).getOrCreate()
# create the context
sc = spark.sparkContext

from functools import reduce

def custom_join(df1, df2):
    return df1.join(df2, 'page_id', 'outer')

if __name__ == '__main__':

    save_path = "/scratch/descourt/processed_data_050223"
    save_file = "pageviews_agg_en_2015-2023.parquet"

    try :
        # Process data
        for args_m, args_y in zip([[1,2,3,4,5,6,7,8,9,10,11,12],
                                   [7,8,9,10,11,12],
                                   [1,2,3]],

                                  [[2016, 2017, 2018, 2019, 2020, 2021, 2022],
                                   [2015],
                                   [2023]]):
            months = [str(m) if m / 10 >= 1 else f"0{m}" for m in args_m]
            dates = [f"{year}-{month}" for year in args_y for month in months]

            dfs = setup_data(years=args_y, months=months)
            df_filt = filter_data(dfs, 'en.wikipedia', dates=dates)

            df_agg = aggregate_data(df_filt)
            df_agg.write.parquet(os.path.join(save_path, f"pageviews_agg_en.wikipedia_{'_'.join(args_y)}.parquet"))

        # Read all again and save
        dfs_path = [os.path.join(save_path, d) for d in os.listdir(save_path)]
        dfs = [spark.read.parquet(df) for df in dfs_path]
        reduce(DataFrame.unionAll, dfs).write.parquet(os.path.join(save_path, save_file))

    except Exception as e:
        print("Exception occured at data processing stage")
        save_path = "/scratch/descourt/processed_data_050123"
        save_file = "pageviews_agg_en.wikipedia_2015_2016_2017_2018_2019_2020_2021_2022_2023.parquet"

    # path
    path = '/scratch/descourt/interm_results/rank_div_all'
    os.makedirs(path, exist_ok=True)

    try :

        # Data - all
        dfs = spark.read.parquet(os.path.join(save_path, save_file))

        # Extract high volume core
        window = Window.partitionBy('date').orderBy('rank')
        df_cutoff = df_rank.withColumn('cum_views', sum('tot_count_views').over(window)) \
            .select(col('date').alias('d'), 'cum_views', 'rank', 'page')
        df_sum = df_rank.groupBy('date').agg(sum('tot_count_views').alias('tot_count_month'))
        df_cutoff = df_cutoff.join(df_sum, df_sum.date == df_cutoff.d) \
            .withColumn('perc_views', col('cum_views') / col('tot_count_month') * 100) \
            .drop('d')
        df_high_volume = df_cutoff.where(df_cutoff.perc_views <= 90) # Empirical

        # consider date per date
        months = [str(m + 1) if (m + 1) / 10 >= 1 else f"0{m + 1}" for m in range(12)]
        dates = ['2015-07', '2015-08', '2015-09', '2015-10', '2015-11', '2015-12'] + \
                [f"{y}-{m}" for y in ['2016', '2017', '2018', '2019', '2020', '2021', '2022'] for m in months] + \
                ['2023-01', '2023-02', '2023-03']

        dfs_divs = []

        for i in tqdm(range(len(dates) - 1)):

            print(f"Processing {dates[i]}-{dates[i+1]}")

            d1 = dates[i]
            d2 = dates[i + 1]
            df_ranks_piv = df_high_volume.where(df_high_volume.date.isin([d1, d2]))\
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

    try :

        paths = [os.path.join(path, f) for f in os.listdir(path)]
        dfs = [spark.read.parquet(p) for p in paths]
        df = reduce(DataFrame.unionAll, dfs)
        df.write.parquet(os.path.join(path, 'rank_turb_div_all.parquet'))

    except Exception as e:
        print("Error occured when saving all RTD")
