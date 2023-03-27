import pandas as pd
from ast import literal_eval
import gzip
import os
import json
from tqdm import tqdm

"""
Temporary file with code to gzip json files
"""

def zip_json(dir_, dir_out):
    ls_dirs = os.listdir(dir_)

    for subdir in tqdm(ls_dirs):
        for file in os.listdir(os.path.join(dir_, subdir)):
            path_in = os.path.join(dir_, subdir, file)
            with open(path_in, 'r') as f:
                data = f.read().replace('\n', ',')
                data = literal_eval(data)
            os.makedirs(os.path.join(dir_out, subdir), exist_ok=True)
            path_out = os.path.join(dir_out, subdir, file)
            with gzip.open(path_out + ".json.gz", 'wt', encoding='utf-8') as f:
                json.dump(data, f)

if __name__ == '__main__':

    dir_ = "/scratch/descourt/wiki_dumps/extracted"
    dir_out = "/scratch/descourt/wiki_dumps/extracted_zip"
    zip_json(dir_, dir_out)