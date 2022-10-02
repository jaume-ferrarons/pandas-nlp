from .models.sentence_model import SentenceModel
from .models.language_model import LanguageModel
from .models.vector_model import VectorModel

from .models.spacy_model import SpacyModel
from .models.fasttext_language_model import FastTextLanguageModel


class ModelManager:
    def __init__(self) -> None:
        self._models = {}

    def get_sentence_model(self, model_name: str = None) -> SentenceModel:
        if model_name not in self._models:
            self._models[model_name] = SpacyModel(model_name)
        return self._models[model_name]

    def get_vector_model(self, model_name: str = None) -> VectorModel:
        if model_name not in self._models:
            self._models[model_name] = SpacyModel(model_name)
        return self._models[model_name]

    def get_language_model(self, model_name: str = None) -> LanguageModel:
        if model_name not in self._models:
            self._models[model_name] = FastTextLanguageModel(model_name)
        return self._models[model_name]
