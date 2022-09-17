from typing import List
import pandas as pd
import spacy
from thinc.types import Floats1d


@pd.api.extensions.register_series_accessor("nlp")
class NLPAccessor:
    def __init__(self, pd_series: pd.Series) -> None:
        self._series = pd_series
        self._validate(pd_series)
        self._model_cache = {}

    @staticmethod
    def _validate(pd_series: pd.Series):
        for v in pd_series:
            if not isinstance(v, str):
                raise TypeError(f"Value {v} is not a string")

    def sentences(self, model: str = "en_core_web_sm") -> List[List[str]]:
        nlp = self._get_model(model)
        return [[sent.text for sent in doc.sents] for doc in nlp.pipe(self._series)]

    def embeddings(self, model: str = "en_core_web_sm") -> List[Floats1d]:
        nlp = self._get_model(model)
        return [doc.vector for doc in nlp.pipe(self._series)]

    def _get_model(self, model: str) -> spacy.language.Language:
        if model not in self._model_cache:
            self._model_cache[model] = spacy.load(model)
        return self._model_cache[model]

    def clearCache(self):
        self._model_cache = {}
