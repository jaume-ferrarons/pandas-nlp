from functools import wraps
from typing import List
import pandas as pd
import spacy

from .fasttext_cli import FastTextCli


def _renameSeriesResult(func):
    @wraps(func)
    def handler(*args, **kwargs):
        original_series: pd.Series = args[0]._series
        result_series: pd.Series = func(*args, **kwargs)
        result_series.rename(original_series.name + "_" + func.__name__, inplace=True)
        return result_series

    return handler


def _handleEmpty(func):
    @wraps(func)
    def handler(*args, **kwargs):
        if len(args[0]._series) == 0:
            return pd.Series([], dtype="object")
        return func(*args, **kwargs)

    return handler


@pd.api.extensions.register_series_accessor("nlp")
class NLPAccessor:
    CLI_BUILDERS = {"fasttext": FastTextCli}

    def __init__(self, pd_series: pd.Series) -> None:
        self._series = pd_series
        self._validate(pd_series)
        self._model_cache = {}
        self._cli_cache = {}

    @staticmethod
    def _validate(pd_series: pd.Series):
        for v in pd_series:
            if not isinstance(v, str):
                raise TypeError(f"Value {v} is not a string")

    @_renameSeriesResult
    @_handleEmpty
    def sentences(self, model: str = "en_core_web_sm") -> pd.Series:
        nlp = self._get_model(model)
        return pd.Series(
            [[sent.text for sent in doc.sents] for doc in nlp.pipe(self._series)]
        )

    @_renameSeriesResult
    @_handleEmpty
    def embedding(self, model: str = "en_core_web_sm") -> pd.Series:
        nlp = self._get_model(model)
        return pd.Series([doc.vector for doc in nlp.pipe(self._series)])

    @_renameSeriesResult
    @_handleEmpty
    def language(self) -> List[str]:
        fasttext = self._get_cli("fasttext")
        return self._series.map(fasttext.language)

    def _get_model(self, model: str) -> spacy.language.Language:
        if model not in self._model_cache:
            self._model_cache[model] = spacy.load(model)
        return self._model_cache[model]

    def _get_cli(self, cli_name: str):
        if cli_name not in self._cli_cache:
            self._cli_cache[cli_name] = self.CLI_BUILDERS[cli_name]()
        return self._cli_cache[cli_name]

    def clearCache(self):
        self._model_cache = {}
