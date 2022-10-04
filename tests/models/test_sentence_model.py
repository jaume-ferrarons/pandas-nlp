from typing import List
import unittest

from pandas_nlp.models.sentence_model import SentenceModel


class DummySentenceModel(SentenceModel):
    def text_sentences(self, text: str) -> List[str]:
        return [sentence.strip() for sentence in text.split(".") if len(sentence) > 0]


class SentenceModelTest(unittest.TestCase):
    _model: DummySentenceModel

    @classmethod
    def setUpClass(cls) -> None:
        cls._model = DummySentenceModel()

    def test_texts_sentences(self):
        sentences = self._model.texts_sentences(["Red. Green. Blue.", "Cars are red."])
        self.assertListEqual(sentences, [["Red", "Green", "Blue"], ["Cars are red"]])
