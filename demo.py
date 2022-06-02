#!/usr/bin/env python
# coding: utf-8
import csv
import gc
import sys
import datetime as dt

from functools import partial
from src.dataset import CsvDataset
from src.preproc import Preprocessor, LangEnum
from src.models import Top2VecW
from tqdm.notebook import tqdm

if len(sys.argv) != 3:
    print('USAGE: demo.py DATASET_PATH OUTPUT_DIR', file=sys.stderr)
    exit(1)

DATASET_PATH = sys.argv[1]
OUTPUT_DIR = sys.argv[2]


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
                       max_size=10000,
                       filter_func=partial(filter_func,
                                           dt_col='date',
                                           time_format=time_format,
                                           year=2018))

with open(OUTPUT_DIR + 'result.csv', 'a') as of:
    writer = csv.writer(of)
    # writer.writerow(['batch_id', 'refdoc_id', 'text', 'date'])

    for i, batch in tqdm(enumerate(ds_generator)):
        gc.collect()

        # event_docs = []
        # event_ids = []
        # event_docs.append([])
        # event_ids.append([])

        frame = list(map(lambda x: x['title'] + ' ' + x['text'], batch))
        doc_iter, _ = preprocessor.preprocess_texts(frame)
        texts = list(map(str, doc_iter))

        try:
            m = Top2VecW(
                documents=texts,
                embedding_model='doc2vec',
                # embedding_model='universal-sentence-encoder-multilingual',
                workers=12,
                tokenizer=crutch_for_top2vec)

            for j in range(m.__model__.get_num_topics()):
                docs, scores, ids = m.__model__.search_documents_by_topic(
                    topic_num=j, num_docs=1)
                for doc, score, doc_id in zip(docs, scores, ids):
                    # print(frame)
                    writer.writerow([
                        f"{i}", batch[doc_id]['id'], frame[doc_id],
                        batch[doc_id]['date']
                    ])
                    # event_docs.append(doc)
                    # event_ids.append(doc_id)
            # models.append(top2vec)
            print(len(batch))
        except Exception as e:
            print(f'Failed to get topics: {e}, len(batch) = {len(batch)}',
                  file=sys.stderr)
            continue
