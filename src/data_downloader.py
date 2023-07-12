import threading
from queue import Queue
import requests
import time

import os
import argparse

"""
Adapted from https://github.com/gesiscss/wiki-download-parse-page-views
"""

def download(url, save_path):
    global q

    """ 
    Downloads file to specified path, if server returns status codes other than 200 url is saved
    """

    f_name = url.split("/")[-1]
    save = save_path + "/" + f_name

    r = requests.get(url, stream=True)
    if r.status_code != 200:
        raise ConnectionError(f"Server did not response correctly: {r.status_code}")
    print("Downloading {}".format(url))

    with open(save, 'wb') as f:
        for chunk in r.iter_content(1024):
            if chunk:
                f.write(chunk)


def threader(save_path):
    global q
    while q.empty() != True:
        # gets an worker from the queue
        worker = q.get()

        # Run the example job with the avail worker in queue (thread)
        download(worker, save_path)

        # completed with the job
        q.task_done()


def start_threads(num_threads, save_path):
    for x in range(num_threads):
        time.sleep(0.5)
        t = threading.Thread(target=threader, args=(save_path,))

        # classifying as a daemon, so they will die when the main dies
        t.daemon = True
        # begins, must come after daemon definition
        t.start()

def main():

    parser = argparse.ArgumentParser(
        description='Wikipedia pageviews downloading'
    )

    parser.add_argument('--nb_threads',
                        type=int,
                        default=3,
                        help='Number of threads for parallelized downloading')
    parser.add_argument('--y',
                        help='year',
                        nargs='+',
                        type=int,
                        default=[2015,2016,2017,2018,2019,2020,2021,2022])
    parser.add_argument('--m',
                        help='months',
                        nargs='+',
                        type=int,
                        default=[1,2,3,4,5,6,7,8,9,10,11,12])
    parser.add_argument('--sp',
                        help='save path',
                        type=str,
                        default='/scratch/descourt/pageviews')
    parser.add_argument('--dtype',
                        help='data type, either pageviews or clickstream data',
                        type=str,
                        choices=['pageviews', 'clickstream'],
                        default='pageviews')
    parser.add_argument('--project',
                        help='Project to download for clickstream data',
                        type=str,
                        default='en')

    args = parser.parse_args()
    os.makedirs(args.sp, exist_ok=True)
    pageviews = (args.dtype == 'pageviews')

    urls = []
    for year in args.y:
        for m in args.m:
            month = str(m)
            if len(str(month)) < 2:
                month = "0" + month
            if pageviews:
                urls.append(
                    f"https://dumps.wikimedia.org/other/pageview_complete/monthly/{year}/{year}-{month}/pageviews-{year}{month}-user.bz2")
            else :
                urls.append(
                    f"https://dumps.wikimedia.org/other/clickstream/{year}-{month}/clickstream-{args.project}wiki-{year}{month}.tsv.gz")

    global q

    start = time.time()
    # TODO because API refuses simultaneous queries for more than 3 dumps ?!
    sliced_urls = [urls[i:i+3] for i in range(0, len(urls), 3)]

    for u in sliced_urls:
        q = Queue()
        for worker in u:
            q.put(worker)

        start_threads(args.nb_threads, args.sp)
        q.join()

    end = time.time()
    duration = end - start

    if duration > 60:
        print("Time used: {} {}".format(duration / 60, "minutes"))
    if duration > 3600:
        print("Time used: {} {}".format(duration / 3600, "hours"))
    else:
        print("Time used: {} {}".format(duration, "seconds"))


if __name__ == '__main__':

    main()

