import unittest

from pandas_nlp.models.spacy_model import SpacyModel


class SpacyModelTest(unittest.TestCase):
    _model: SpacyModel

    @classmethod
    def setUpClass(cls) -> None:
        cls._model = SpacyModel("en_core_web_md")

    def test_vector(self):
        vector = self._model.text_vector("this is me")
        self.assertEqual(len(vector), 300)

    def test_vectors(self):
        vectors = self._model.texts_vectors(["this is me"])
        self.assertEqual(len(vectors), 1)
        self.assertEqual(len(vectors[0]), 300)

    def test_sentence(self):
        sentences = self._model.text_sentences("Hello. It's cool!")
        self.assertListEqual(sentences, ["Hello.", "It's cool!"])

    def test_sentences(self):
        sentences = self._model.texts_sentences(["Hello. It's cool!"])
        self.assertEqual(len(sentences), 1)
        self.assertListEqual(sentences[0], ["Hello.", "It's cool!"])
