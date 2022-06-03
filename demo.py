#!/usr/bin/env python
# coding: utf-8
import csv
import gc
import sys
import datetime as dt

from functools import partial
from event_detection.dataset import CsvDataset
from event_detection.preproc import Preprocessor, LangEnum
from event_detection.extractor import EventExtractor
from tqdm.notebook import tqdm

if len(sys.argv) != 3:
    print('USAGE: demo.py DATASET_PATH OUTPUT_PATH', file=sys.stderr)
    exit(1)

DATASET_PATH = sys.argv[1]
OUTPUT_PATH = sys.argv[2]


def crutch_for_top2vec(doc):
    return doc.split()


def filter_func(x: dict, dt_col: str, time_format: str, year: int) -> bool:
    cur_time = dt.datetime.strptime(x[dt_col], time_format).date()
    return cur_time.year == year


dataset = CsvDataset(DATASET_PATH)
preprocessor = Preprocessor(language=LangEnum.RU, tokenize_ents=False)

time_format = "%Y-%m-%d %H:%M:%S"
ds_generator = dataset(time_window=dt.timedelta(days=2),
                       time_col="date",
                       columns=['title', 'id', 'text', 'date'],
                       time_format=time_format,
                       max_size=1000,
                       filter_func=partial(filter_func,
                                           dt_col='date',
                                           time_format=time_format,
                                           year=2015))

extractor = EventExtractor(embedding_model='doc2vec')

with open(OUTPUT_PATH, 'w') as of:
    writer = csv.writer(of)
    # writer.writerow(['batch_id', 'refdoc_id', 'text', 'date'])

    for i, batch in tqdm(enumerate(ds_generator)):
        gc.collect()

        for doc in batch:
            doc['text'] = doc['title'] + '. ' + doc['text']
            doc.pop('title', None)

        try:
            events = extractor.extract_events(batch, num_workers=6)
            print(events)

        except Exception as e:
            print(f'Failed to get topics: {e}, len(batch) = {len(batch)}',
                  file=sys.stderr)
            continue
