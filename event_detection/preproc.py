import spacy
import re
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.language import Language
from spacy.pipe_analysis import Doc
from spacy.util import compile_infix_regex
from gensim.corpora.dictionary import Dictionary
from string import whitespace
from itertools import tee
from enum import Enum
from os import cpu_count
from typing import Iterable


class LangEnum(Enum):
    """
    Enum to represent supported language codes
    """
    EN = 0
    RU = 1


@spacy.Language.component(name='merge_entities')
def merge_entities(doc):
    with doc.retokenize() as retokenizer:
        for ent in doc.ents:
            retokenizer.merge(
                doc[ent.start:ent.end],
                attrs={'LEMMA': re.sub(r'\s+', '_', str(ent.lemma_))})
    return doc


class Preprocessor:
    """
    Use this class to encapsulate Spacy models, Gensim stuff and everything
    else needed for text preprocessing.
    """

    def __init__(self,
                 language: LangEnum = 0,
                 stop_words: Iterable[str] = None,
                 tokenize_ents: bool = True,
                 workers: int = cpu_count()):
        # Preload ready to use spacy model (tokenizer, lemmatizer, etc)
        if language == LangEnum.EN:
            self.nlp: Language = spacy.load('en_core_web_sm')
        elif language == LangEnum.RU:
            self.nlp: Language = spacy.load('ru_core_news_md')
        else:
            raise NotImplementedError('Only Russian and English '
                                      'languages are supported at the moment')

        # Wheter or not to tokenize detected named entities
        self.tokenize_ents = tokenize_ents
        self.workers = workers

        # Modify tokenizer infix patterns
        infixes = (
            LIST_ELLIPSES + LIST_ICONS + [
                r"(?<=[0-9])[+\-\*^](?=[0-9-])",
                r"(?<=[{al}{q}])\.(?=[{au}{q}])".format(
                    al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES),
                r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
                # r"(?<=[{a}])(?:{h})(?=[{a}])".format(a=ALPHA, h=HYPHENS),
                r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
            ])

        infix_re = compile_infix_regex(infixes)
        self.nlp.tokenizer.infix_finditer = infix_re.finditer

        # Update the built-in stopwords list
        if stop_words is not None:
            self.update_stopwords(stop_words)

        if not tokenize_ents:
            self.nlp.add_pipe('merge_entities', last=True)

    def filter_tokens(self, docs: Iterable[Doc]) -> Iterable[tuple[Doc, str]]:
        for doc in docs:
            tokens = []

            for token in doc:
                if not (token.is_stop or token.is_punct or token.like_email
                        or token.like_url or token.is_space
                        or token.is_currency or token.like_num or
                        token.lemma_.lower() in self.nlp.Defaults.stop_words):
                    tokens.append(token.lemma_.lower())

            yield (doc, ' '.join(tokens))

    def update_stopwords(self, stop_words: Iterable[str]) -> None:
        """
        Update built-in spacy language model stopwords list

        :param stop_words: Iterable of strings - target stopwords
        :return: None
        """
        self.nlp.Defaults.stop_words.update(stop_words)
        for word in self.nlp.Defaults.stop_words:
            lexeme = self.nlp.vocab[word]
            lexeme.is_stop = True

    def preprocess_texts(self,
                         data: Iterable[str]) -> (Iterable[Doc], Dictionary):
        """
        Get preprocessed texts

        :param data: iterable of strings
                     (each string is considered to be a single document)
        :return: preprocessed documents and
                a gensim Dictionary of the given docs
        """
        docs = self.__get_preprocessed_docs__(data)
        docs, docs_iter_copy = tee(docs)
        return docs, Dictionary(
            map(lambda x: [y.lemma_ for y in x], docs_iter_copy))

    def __get_preprocessed_docs__(self, data: Iterable[str]):
        """
        Helper function to generate new docs using spacy Language.pipe()

        :param data: iterable of strings (1 string = 1 doc)
        :return: spacy Document generator
        """
        docs = self.nlp.pipe(data, n_process=self.workers)
        for doc in docs:
            yield doc
