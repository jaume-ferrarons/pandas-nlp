from functools import wraps
from typing import List
import pandas as pd
from pandas.api.extensions import register_series_accessor

from pandas_nlp.model_manager import ModelManager


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


def register():
    """Registers pandas_nlp `nlp` accessor for pandas series"""
    register_series_accessor("nlp")(NLPAccessor)


class NLPAccessor:
    def __init__(self, pd_series: pd.Series) -> None:
        self._series = pd_series
        self._validate(pd_series)
        self._mm = ModelManager()

    @staticmethod
    def _validate(pd_series: pd.Series):
        for v in pd_series:
            if not isinstance(v, str):
                raise TypeError(f"Value {v} is not a string")

    @_renameSeriesResult
    @_handleEmpty
    def sentences(self, model: str = "en_core_web_md") -> pd.Series:
        return pd.Series(
            self._mm.get_sentence_model(model).texts_sentences(self._series)
        )

    @_renameSeriesResult
    @_handleEmpty
    def embedding(self, model: str = "en_core_web_md") -> pd.Series:
        return pd.Series(self._mm.get_vector_model(model).texts_vectors(self._series))

    @_renameSeriesResult
    @_handleEmpty
    def closest(self, labels: List[str], model: str = "en_core_web_md") -> pd.Series:
        return pd.Series(
            self._mm.get_vector_model(model_name=model).closest(self._series, labels)
        )

    @_renameSeriesResult
    @_handleEmpty
    def language(
        self, confidence: bool = False, model: str = "lid.176.ftz"
    ) -> List[str]:
        languages = self._mm.get_language_model(model).languages(self._series)
        result = [
            {"language": entry[0], "confidence": entry[1]} if confidence else entry[0]
            for entry in languages
        ]
        return pd.Series(result)
