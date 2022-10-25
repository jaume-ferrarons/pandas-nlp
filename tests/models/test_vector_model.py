import unittest

import numpy as np

from pandas_nlp.models.vector_model import VectorModel
from tests.utils import register_pandas_nlp_if_not_available


class DummyVectorModel(VectorModel):
    def text_vector(self, text: str) -> np.ndarray:
        return np.array([len(text), -len(text)])


class SentenceModelTest(unittest.TestCase):
    _model: DummyVectorModel

    @classmethod
    def setUpClass(cls) -> None:
        register_pandas_nlp_if_not_available()
        cls._model = DummyVectorModel()

    def test_texts_vectors(self):
        sentences = self._model.texts_vectors(["Red. Green. Blue.", "Cars are red."])
        self.assertEqual(len(sentences), 2)
        self.assertTrue((sentences[0] == [17, -17]).all())
        self.assertTrue((sentences[1] == [13, -13]).all())
