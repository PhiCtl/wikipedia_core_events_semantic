# A longitudinal characterization of Wikipedia's core and tail


## Set-up

```
conda env create -f setup/conda.yml
conda activate wiki_sem
```



## Repository overall structure

```bash
├── notebooks         	# Notebooks for analyses 
│   ├── Data_presentation.ipynb
│   ├── ...
├── data         		# Additional data used for plotting 
│   ├── WikiStats_views_En.csv
│   ├── ...
├── src         		# Python scripts and helpers
│   ├── data_aggregation.py 
│   ├── ...
├── README.md
├── setup        		# Setup scripts
│   ├── ...

```

## Usage

### Data processing and extraction

#### Raw data

Raw data is located in `/scratch/descourt/raw_data`

All commands have to be run in the `src` folder.  

* Downloading of raw monthly page dumps. The functions have to be adapted to deal with other time scales such as daily or hourly views, but it requires few changes. Run the following command `python data_downloader.py ` and then `python data_downloader.py -- 2023 --m 1 2 3`. Note that we'll be only processing data from July 2015.
* Data processing with redirect matching according to heuristics described in the thesis (example for English edition). Uncomment `automated_main()` in the main and run`python data_aggregation.py --project en`. The functions have to be adapted to deal with other time scales such as daily or hourly views, but it requires few changes.
* Additional redirect matching. Uncomment `match_missing_ids(False)` and run  `python data_aggregation.py --project en`.
* Matching page identifiers and names over months to correct for page moves. Note that it represents a very tiny fraction of the data. Uncomment `match_over_months()` and run the command  `python data_aggregation.py --project en --date date_of_your_choice` .

The above processing is available for both English and French editions.  All processed data can already be found under 

```bash
├── processed_data
│   ├── en
│   	├── pageviews_en_2015-2023.parquet # Processing and redirect matching were already performed here
│   	├── pageviews_en_articles_ev_{"date".parquet} # Matched page identifiers and names over months
│   ├── fr
│   	├── ... # Same datasets of interest
│   ├── multieditions
│   	├── ... # Datasets for tail / non tail experiment
```



#### Metadata

Raw metadata are located under `/scratch/descourt/metadata`. To process them for specific editions, please change the `main()` of `src/data_enrich_processing.py`.

```bash
├── akhils_metadata         # Metadata provided by Akhil for all editions 
│   ├── wiki_nodes_bsdk_phili_2022-11_{"", "en", "fr"}.parquet # Some nodes properties discussed in the thesis as of Nov 22
│   ├── wiki_nodes_bios_bsdk_phili_2022-11.parquet             # Same properties as above for biographies articles + is_woman attribute
│   ├── wiki_nodes_topics_2022-09.parquet             # Topics for all editions as of Sept 2022
│   ├── wiki_nodes_2022-11.parquet # In and out degree, is_orphan properties for all editions as of Nov 2022
├── quality
│   ├── ORES_quality_en_March21.json.gz # ORES quality scores for En edition as of March 2021, from dlab server. unused
│   ├── ORES_quality_en_March21.parquet # Same as above, processed
├── semantic_embeddings
│   ├── en
│   	├── article-description-embeddings_enwiki-20210401-fasttext.pickle # Semantic embeddings from dlab server raw, en edition
│   	├── embeddings-20210401-norm.parquet # Semantic embeddings from dlab server processed and normalized, en edition
│   ├── fr ... # Same as above for French edition
├── topics
│   ├── topics_en
|		├── topics_enwiki.tsv.zip # Raw OREs topic predictions for En edition from dlab server
|		├── topics-enwiki-20230320-parsed.parquet # Above dataset processed

```

### Report analysis and figures

* Data general presentation and statistics, hinge method and common pages stability analyses and figures are available by running the notebook `Data_presentation.ipynb`
* Characterisation of the core and the tail analyses and figures can be reproduced by running the notebook `Enrich_dataset.ipynb`
* Rank turbulence divergence : first run in `src` the command `python make_and_plot.py --mode rtd --memory 120 ` and  `python make_and_plot.py --mode alpha --memory 120 ` . The files of interest would be saved under `/scratch/descourt/plots/thesis/RTD_all.parquet` and `/scratch/descourt/plots/thesis/divs_alphas.csv.gzip` . Then run the notebook `Rank_stability_analysis.ipynb`. 
* Volume dynamics are available when running through the following notebook `Volume_dynamics.ipynb`.

## Additional work

* Work on False positive is available in `Data_presentation.ipynb`
* Work on pairs matching is available in `Pairs.ipynb`. Pairs were extracted running the command `python make_pairs.py --memory 120`. Note that core versus tail estimation is biased because we filtered out all pages containing a colon to filter main namespace pages, and this is wrong. In future work, another dataset should be considered such as hourly or daily pageviews, in which it becomes easier to filter main namespace. 

