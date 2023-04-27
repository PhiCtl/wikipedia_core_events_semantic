import os
import pyspark
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark import SparkContext
from tqdm import tqdm

from volumes_extraction import *

os.environ["JAVA_HOME"] = "/lib/jvm/java-11-openjdk-amd64"
conf = pyspark.SparkConf().setMaster("local[*]").setAll([
                                   ('spark.driver.memory','60G'),
                                   ('spark.executor.memory', '60G'),
                                   ('spark.driver.maxResultSize', '0'),
                                    ('spark.executor.cores', '5')
                                  ])
# create the session
spark = SparkSession.builder.config(conf=conf).getOrCreate()
# create the context
sc = spark.sparkContext


class DataExplorer:

    def __init__(self):
        self.raw_data = None
        self.stats = None

    def load_data(self, path, edition):
        self.raw_data = spark.read.parquet(f"{path}/pageviews_agg_{edition}_2015_2016_2017_2018_2019_2020_2021_2022.parquet")

    def compute_stats(self, normalized=False):
        self.stats = self.raw_data.groupBy('date').agg(max('tot_count_views').alias('max'), \
                                              min('tot_count_views').alias('min'), \
                                              percentile_approx('tot_count_views', 0.5).alias('med'), \
                                              percentile_approx('tot_count_views', 0.25).alias('25%'), \
                                              percentile_approx('tot_count_views', 0.75).alias('75%'), \
                                              percentile_approx('tot_count_views', 0.9).alias('90%'), \
                                              avg('tot_count_views').alias('avg'), \
                                              sum('tot_count_views').alias('sum'),
                                              count("*").alias('nb_pages')).toPandas()
        if normalized:
            self.stats.rename({col : col + '_perc' for col in self.stats\
                                if col not in ['max', 'date', 'sum', 'nb_pages'] }, axis=1, inplace=True)
            cols_norm = [col for col in self.stats.columns if col not in ['max', 'date', 'sum', 'nb_pages']]
            self.stats[cols_norm] = self.stats[cols_norm].div(self.stats['max'], axis=0)

        self.stats.head(self.stats.shape[0])

    def compute_monthly_hist(self, dates):
        # TODO faster histogram plot
        df_plot = self.stats.where(self.stats.date.isin(*dates)).select('date', 'tot_count_views')

    # TODO fulfill

if __name__ == '__main__':

    dfs = spark.read.parquet(
        "/scratch/descourt/processed_data/pageviews_agg_en.wikipedia_2015_2016_2017_2018_2019_2020_2021_2022.parquet")
    # Extract monthly high volume pages
    df_rank = compute_ranks_bins(dfs, slicing=1000, lim=int(9 * 1e7))
    df_cutoff = extract_volumes(df_rank).where(
        "perc_views <= '90.5'")  # Empirical, used for extracting rank divergences on high volume
    total_dates = [d['date'] for d in df_cutoff.select('date').distinct().sort(asc('date')).collect()]

    perc_seensofar_overlap = []

    for i in tqdm(np.arange(0, len(total_dates[:75]), 3)):
        df_i = df_cutoff.where(df_cutoff.date.isin(total_dates[:i + 3])).groupBy('page').agg(
            count('*').alias('nb_occurrences'))
        df_i = df_cutoff.where(df_cutoff.date.isin(total_dates[:i + 3])).select('date', 'page').join(df_i, 'page')
        # Percentage with respect to all pages seen so far
        nb_overlap_i = df_i.filter(f'nb_occurrences == "{i + 3}"').select('page').distinct().count()
        nb_seen_so_far = df_i.select('page').distinct().count()
        perc_seensofar_overlap.append(nb_overlap_i / nb_seen_so_far * 100)

    perc_seensofar_overlap = np.array(perc_seensofar_overlap)

    plt.figure(figsize=(15, 5))

    plt.plot(perc_seensofar_overlap)
    plt.xticks(range(15), labels=total_dates[:75][::3], rotation=45)
    plt.ylabel('Percentage of stable pages over all pages seen so far')
    plt.xlabel('date')

    plt.savefig("/home/descourt/interm_results/overlap_evolution.png")


