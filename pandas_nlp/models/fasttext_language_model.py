from typing import Tuple
import fasttext
from fasttext.FastText import _FastText


from .language_model import LanguageModel
from ..cache import get_remote_file

# Remove useless warning
fasttext.FastText.eprint = lambda x: None


class FastTextLanguageModel(LanguageModel):
    FASTTEXT_MODELS = {
        "lid.176.ftz": "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
    }

    def __init__(self, model_name: str = "lid.176.ftz") -> None:
        super().__init__()
        self._model = self._load_model(model_name)

    def language(self, text: str) -> Tuple[str, float]:
        """Returns the language with confidence of the text"""
        (langs, scores) = self._model.predict(text)
        language = langs[0][-2:]
        score = scores[0]
        return language, score

    def _load_model(self, model_name: str) -> _FastText:
        path = get_remote_file("fasttext", model_name, self.FASTTEXT_MODELS[model_name])
        return fasttext.load_model(str(path))
