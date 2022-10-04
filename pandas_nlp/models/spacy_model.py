import spacy
import numpy as np
from typing import List

from .vector_model import VectorModel
from .sentence_model import SentenceModel


class SpacyModel(VectorModel, SentenceModel):
    def __init__(self, model_name: str = "en_core_web_md") -> None:
        super().__init__()
        self._model = spacy.load(model_name)

    def text_vector(self, text: str) -> np.ndarray:
        """Returns the embedding of the provided text"""
        doc = self._model(text)
        return np.array(doc.vector)

    def texts_vectors(self, texts: List[str]) -> List[np.ndarray]:
        return [np.array(doc.vector) for doc in self._model.pipe(texts)]

    def text_sentences(self, text: str) -> List[str]:
        doc = self._model(text)
        return [sent.text for sent in doc.sents]

    def texts_sentences(self, texts: List[str]) -> List[List[str]]:
        return [[sent.text for sent in doc.sents] for doc in self._model.pipe(texts)]
