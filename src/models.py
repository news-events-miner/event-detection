"""
This module contains generic model wrapper and some
default implementations (LDA, NMF).
"""
from abc import ABC, abstractmethod
from itertools import tee
from typing import Any, Optional, Iterable, Tuple
from gensim.models import LdaModel, LdaMulticore
from functools import partial
from top2vec import Top2Vec


class GenericModel(ABC):
    """
    GenericModel class defines a basic interface
    for all other models. You need to wrap up your model (
    e.g. a gensim model) in such class and implement 2 methods:

    1. fit() which gets data for model training/fitting.
    2. update() to update the model.
    3. get_topics() to return the extracted topics.
    """

    @abstractmethod
    def fit(self, data: Any, *args, **kwargs) -> None:
        """
        Fit freshly created model instance.

        :param data: train data (in any format supported by your model)
        :param args: additional positional args that you need
        :param kwargs: keyword args for your model
        :return: None
        """
        pass

    @abstractmethod
    def update(self, data: Any, *args, **kwargs) -> None:
        """
        Update the model with new data

        :param data: new data to re-fit on
        :param args: additional args if you need this
        :param kwargs: additional kwargs if you need this
        :return: None
        """
        pass

    @abstractmethod
    def get_topics(self,
                   docs: Optional[Iterable[Any]] = None,
                   *args, **kwargs) -> Iterable[Tuple[int, Tuple[str, float]]]:
        """
        Get topics extracted from docs

        :param docs: new document collection,
                     if None (default) returns topics extracted from the
                     latest model state
        :param args: additional positional arguments
        :param kwargs: additional keyword arguments
        :return: topics - any Iterable of Tuple[id, Tuple[word, prob]]
        """
        pass


class LDA(GenericModel):
    """
    Wrapper for Gensim LdaModel and LdaMulticore
    """

    def __init__(self, *args, **kwargs):
        """
        All provided arguments will be passed to LdaModel or
        LdaMulticore constructors (the latter in case 'workers'
        is present in keyword arguments)

        :param args: positional arguments to initialize model with
        :param kwargs: keyword arguments to pass to model constructor
        """
        if 'workers' in kwargs.keys():
            self.__model__ = LdaMulticore(*args, **kwargs)
        else:
            self.__model__ = LdaModel(*args, **kwargs)

    def fit(self, data: Any, *args, **kwargs):
        # Actually, I think there is no need for this as
        # we can simply use update() for uninitialized model
        self.__model__.update(corpus=data, *args, **kwargs)

    def update(self, data: Any, *args, **kwargs):
        self.__model__.update(corpus=data, *args, **kwargs)

    def get_topics(self, docs: Optional[Iterable[Any]] = None, *args, **kwargs):
        if docs is None:
            topics = self.__model__.show_topics(formatted=False,
                                                *args, **kwargs)
        else:
            topics = map(
                partial(self.__model__.get_document_topics,
                        per_word_topics=True),
                docs
            )
        topics, t_copy, t_copy_1 = tee(topics, 3)

        ids = map(lambda x: x[0], topics)
        words = map(lambda x: x[1], t_copy)
        words = map(lambda x: list(zip(*x))[0], words)
        scores = map(lambda x: x[1], t_copy_1)
        scores = map(lambda x: list(zip(*x))[1], scores)

        topics = zip(ids, zip(words, scores))

        return topics


class NMF(GenericModel):
    """
    Wrapper for Gensim Non-negative matrix factorization topic model
    """

    def __init__(self, *args, **kwargs):
        pass

    def fit(self, data: Any, *args, **kwargs):
        pass

    def update(self, data: Any, *args, **kwargs):
        pass

    def get_topics(self, docs: Optional[Any] = None, *args, **kwargs):
        pass


class Top2VecW(GenericModel):
    """Wrapper for top2vec model"""

    def __init__(self, *args, **kwargs):
        """Initialize Top2Vec with the given args"""
        self.__model__ = Top2Vec(*args, **kwargs)

    def fit(self, data: Any, *args, **kwargs) -> None:
        print("Top2Vec API requires to pass docs in object constructor. "
              "Please, pass them with `documents` constructor parameter.")
        pass

    def update(self, data: Any, *args, **kwargs) -> None:
        self.__model__.add_documents(data, *args, **kwargs)

    def get_topics(self, docs: Optional[Iterable[Any]] = None, *args,
                   **kwargs) -> Iterable[Tuple[int, Tuple[str, float]]]:
        if docs is not None:
            # return self.__model__.get_documents_topics()
            raise NotImplementedError("Currently, the only possible way to "
                                      "infer is to update model or retrain "
                                      "with new docs.")
        else:
            topic_words, word_scores, topic_nums = self.__model__.get_topics(
                *args, **kwargs
            )
            topic_words = [[str(t) for t in w.tolist()] for w in topic_words]
            return zip(topic_nums, zip(topic_words, word_scores))
