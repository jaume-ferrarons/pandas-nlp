import unittest

import numpy as np

from pandas_nlp.models.vector_model import VectorModel


class DummyVectorModel(VectorModel):
    def text_vector(self, text: str) -> np.array:
        return np.array([len(text), -len(text)])


class SentenceModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._model = DummyVectorModel()

    def test_texts_vectors(self):
        sentences = self._model.texts_vectors(["Red. Green. Blue.", "Cars are red."])
        self.assertEqual(len(sentences), 2)
        self.assertTrue((sentences[0] == [17, -17]).all())
        self.assertTrue((sentences[1] == [13, -13]).all())
