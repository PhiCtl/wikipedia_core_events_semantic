import pandas as pd
from ast import literal_eval

if __name__ == '__main__':

    df_topics = pd.read_csv("/home/descourt/topic_embeddings/topics_enwiki.tsv.zip",
                            sep='\t')
    df_topics['topics'] = df_topics['topics'].apply(
        lambda st: [literal_eval(s.strip())['topic'] for s in st[1:-1].split("\n")])
    df_topics['page_title'] = df_topics['page_title'].apply(lambda p: p.lower() if isinstance(p, str) else p)
    df_topics.to_csv('/home/descourt/topic_embeddings/topics-enwiki-20230320.csv.gzip', compression='gzip')