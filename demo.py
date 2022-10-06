#!/usr/bin/env python
# coding: utf-8
import os
import gc
import sys
import datetime as dt

from event_detection.dataset import CsvDataset
from event_detection.preproc import Preprocessor, LangEnum
from event_detection.extractor import EventExtractor
from tqdm.notebook import tqdm

if len(sys.argv) != 3:
    print('USAGE: demo.py DATASET_PATH OUTPUT_PATH', file=sys.stderr)
    exit(1)

DATASET_PATH = sys.argv[1]
OUTPUT_PATH = sys.argv[2]


def filter_func(x: dict, dt_col: str, time_format: str, year: int) -> bool:
    cur_time = dt.datetime.strptime(x[dt_col], time_format).date()
    return cur_time.year == year


dataset = CsvDataset(DATASET_PATH)
preprocessor = Preprocessor(language=LangEnum.RU, tokenize_ents=False)

time_format = "%Y-%m-%d %H:%M:%S"
ds_generator = dataset(
    time_window=dt.timedelta(days=3),
    time_col="date",
    columns=['title', 'id', 'text', 'date'],
    time_format=time_format,
    # max_size=1000,
    # filter_func=partial(filter_func,
    #                     dt_col='date',
    #                     time_format=time_format,
    #                     year=2015))
)

extractor = EventExtractor(embedding_model='doc2vec')

DB_HOST = os.environ.get('DB_HOST')
DB_PORT = os.environ.get('DB_PORT')
DB_USER = os.environ.get('DB_USER')
DB_PASSWORD = os.environ.get('DB_PASSWORD')
DB_NAME = os.environ.get('DB_NAME')

DB_URL = "http://{user}:{password}@{host}:{port}/{name}"

with open(OUTPUT_PATH, 'w') as of:
    # writer = csv.writer(of)
    # writer.writerow(['batch_id', 'refdoc_id', 'text', 'date'])
    # writer.writerow(['batch_id', 'text', 'date'])

    of.write("[\n")

    for i, batch in tqdm(enumerate(ds_generator)):
        gc.collect()

        # for doc in batch:
        #     doc['text'] = doc['title'] + '. ' + doc['text']
        #     doc.pop('title', None)

        events = extractor.extract_events(batch, num_workers=3)
        import json
        end = ""
        for event in events:
            if event['doc_scores'][0] > 0.8 and \
                    event['kw_scores'][0] > 0.8:
                if len(end) > 0:
                    of.write(end)

                of.write(json.dumps(event))
                end = ",\n"

    of.write("\n]")
    # writer.writerow(
    #             (i, batch[event['doc_ids'][0]]['title'], event['date']))
    # payload = event

    # req = requests.post(DB_URL.format(user=DB_USER,
    #                                   password=DB_PASSWORD,
    #                                   host=DB_HOST,
    #                                   port=DB_PORT,
    #                                   name=DB_NAME),
    #                     json=payload)
    # pprint(events)
    # try:
    #     events = extractor.extract_events(batch, num_workers=6)
    #     pprint(events)

    # except Exception as e:
    #     print(f'Failed to get topics: {e}, len(batch) = {len(batch)}',
    #           file=sys.stderr)
    #     continue
