import fasttext
from fasttext.FastText import _FastText

from .cache import get_remote_file

# Remove useless warning
fasttext.FastText.eprint = lambda x: None


class FastTextCli:
    FASTTEXT_MODELS = {
        "lid.176.ftz": "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
    }

    def __init__(self) -> None:
        self._model_cache = {}

    def language(self, text: str, confidence: bool = False) -> str:
        (langs, scores) = self._get_model("lid.176.ftz").predict(text)
        language = langs[0][-2:]
        score = scores[0]
        if confidence:
            return {"language": language, "confidence": score}
        return language

    def _get_model(self, model_name: str) -> _FastText:
        if model_name not in self._model_cache:
            self._model_cache[model_name] = self._load_model(model_name)
        return self._model_cache[model_name]

    def _load_model(self, model_name: str) -> _FastText:
        path = get_remote_file("fasttext", model_name, self.FASTTEXT_MODELS[model_name])
        return fasttext.load_model(str(path))
