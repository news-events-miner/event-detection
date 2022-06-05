import datetime as dt
from .models import Top2VecW
from .preproc import Preprocessor, LangEnum
from typing import Callable, Optional, List, Union


class EventExtractor:
    """
    Wrapper around Top2Vec to extract events
    from news documents.
    """

    def __init__(self,
                 embedding_model: str = 'doc2vec',
                 lang: LangEnum = LangEnum.RU,
                 tokenize_ents: Optional[bool] = False,
                 tokenizer: Optional[Callable[str, List[str]]] = None):
        self.embedding_model = embedding_model
        self.language = lang
        self.preproc = Preprocessor(language=lang, tokenize_ents=tokenize_ents)
        self.tokenize_ents = tokenize_ents

        if tokenizer is None:
            # Stub for convenience
            tokenizer = (lambda x: x.split())
        self.tokenizer = tokenizer

    def __getstate__(self):
        attrs = {
            'embedding_model': self.embedding_model,
            'language': self.language,
            'tokenizer': self.tokenizer,
            'tokenize_ents': self.tokenize_ents,
        }
        return attrs

    def __setstate__(self, state):
        self.embedding_model = state.embedding_model
        self.tokenizer = state.tokenizer
        self.language = state.language
        self.tokenize_ents = state.tokenize_ents

        self.preproc = Preprocessor(language=self.language,
                                    tokenize_ents=self.tokenize_ents)

    def extract_events(self,
                       documents: List[dict[str, Union[int, dt.date, str]]],
                       max_topics: Optional[int] = None,
                       max_docs: Optional[int] = 100,
                       num_workers: Optional[int] = 1,
                       reduce_topics: Optional[bool] = False) -> List[dict]:
        frame = map(lambda x: x['text'], documents)
        doc_iter, _ = self.preproc.preprocess_texts(frame)

        filtered = self.preproc.filter_tokens(doc_iter)
        texts = list(map(lambda x: x[1], filtered))

        model = Top2VecW(documents=texts,
                         embedding_model=self.embedding_model,
                         keep_documents=False,
                         workers=num_workers,
                         tokenizer=self.tokenizer)
        events = []

        sizes, _ = model.__model__.get_topic_sizes()

        for j in range(model.__model__.get_num_topics()):
            scores, ids = model.__model__.search_documents_by_topic(
                topic_num=j, num_docs=sizes[j])

            events.append({
                'date': None,
                'doc_ids': [],
                'doc_scores': [],
                'doc_texts': [],
                'place': None,
                'keywords': [],
                'kw_scores': [],
            })

            first = True
            for score, doc_id in zip(scores, ids):
                if first:
                    first = False
                    events[j]['date'] = documents[doc_id]['date']
                events[j]['doc_ids'].append(int(doc_id))
                events[j]['doc_scores'].append(float(score))
                events[j]['doc_texts'].append(str(documents[doc_id]['text']))

        topic_words, topic_scores, nums = model.__model__.get_topics(
            max_topics)

        for topic_id in nums:
            for word, score in zip(topic_words[topic_id],
                                   topic_scores[topic_id]):
                events[topic_id]['keywords'].append(str(word))
                events[topic_id]['kw_scores'].append(float(score))

        return events
