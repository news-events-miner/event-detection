#!/usr/bin/env python
# coding: utf-8
import csv
import gc
import sys
# import logging
import datetime as dt

from csv import DictReader
from typing import List, Callable, Optional
from models import Top2VecW
from functools import partial
from preproc import Preprocessor, LangEnum
from tqdm.notebook import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration

# logging.disable(logging.WARNING)

DATASET_DIR = "/run/media/mkls/Media/Загрузки"
output_dir = "/run/media/mkls/Media/ml/coursework2/events-topics/"


class CsvDataset:

    def __init__(self, fname: str):
        self.fname = fname

    def __call__(self,
                 batch_size: int = 0,
                 time_col: str = "date",
                 time_window: Optional[dt.timedelta] = None,
                 columns: Optional[List[str]] = None,
                 time_format: Optional[str] = None,
                 drop_last: bool = True,
                 text_column: str = "text",
                 max_size: int = 0,
                 filter_func: Optional[Callable[dict, bool]] = None):
        batch = []
        start_time = None

        assert (time_window is not None and time_col is not None
                and time_format is not None) or batch_size != 0

        with open(self.fname, "r") as fd:
            reader = DictReader(fd)
            for line in reader:
                cur_time = dt.datetime.strptime(line[time_col],
                                                time_format).date()

                if start_time is None:
                    batch = []
                    start_time = cur_time

                if len(batch) > 0 and (
                    (time_window is not None
                     and cur_time > start_time + time_window) or
                    (batch_size != 0 and len(batch) == batch_size) or
                    (max_size > 0 and len(batch) == max_size)):
                    start_time = None
                    yield batch
                else:
                    if len(batch) > 0 and line[text_column] != batch[-1][
                            text_column] or len(batch) == 0:
                        if filter_func is not None and filter_func(
                                line) or filter_func is None:
                            batch.append({key: line[key] for key in columns})
        if len(batch) > 0 and not drop_last:
            yield batch
        raise StopIteration


def summarize(text: str, model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    input_ids = tokenizer([text],
                          add_special_tokens=True,
                          padding="max_length",
                          truncation=True,
                          max_length=200,
                          return_tensors="pt")["input_ids"]

    output_ids = model.generate(input_ids=input_ids,
                                no_repeat_ngram_size=3,
                                max_length=128,
                                num_beams=20,
                                early_stopping=True)[0]

    summary = tokenizer.decode(output_ids, skip_special_tokens=True)
    return summary


# dataset = CsvDataset(DATASET_DIR + '/lenta-ru-news.csv')
dataset = CsvDataset(DATASET_DIR + '/aggr_articles_consolidated_2010-2021.csv')

preprocessor = Preprocessor(language=LangEnum.RU, tokenize_ents=False)


def crutch_for_top2vec(doc):
    return doc.split()


def filter_func(x: dict, dt_col: str, time_format: str, year: int) -> bool:
    cur_time = dt.datetime.strptime(x[dt_col], time_format).date()
    return cur_time.year == year


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
# model_name = "IlyaGusev/rut5_base_headline_gen_telegram"
# model_name = "IlyaGusev/rut5_base_sum_gazeta"

with open(output_dir + 'result.csv', 'a') as of:
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
                # embedding_model='doc2vec',
                embedding_model='universal-sentence-encoder-multilingual',
                # embedding_model='paraphrase-multilingual-MiniLM-L12-v2',
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
