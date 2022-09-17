import unittest
import pandas as pd

import pandas_nlp


class SeriesAccessorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._df_words = pd.DataFrame(
            {"id": [1, 2, 3], "text": ["cat", "dog", "violin"]}
        )
        cls._df_sentences = pd.DataFrame(
            {"id": [1, 2, 3], "text": ["", "Hello, how are you?", "Code. Sleep. Eat"]}
        )

    def test_sentences(self):
        self.assertListEqual(
            self._df_sentences.text.nlp.sentences(),
            [[], ["Hello, how are you?"], ["Code.", "Sleep.", "Eat"]],
        )

    def test_embeddings(self):
        embeddings = self._df_words.text.nlp.embeddings()
        self.assertEqual(len(embeddings), 3, "Incorrect number of embeddings returned")
        for embedding in embeddings:
            self.assertEqual(len(embedding), 96)

    def test_nlp_on_not_str(self):
        with self.assertRaises(TypeError) as info:
            self._df_words.id.nlp.sentences()
        self.assertEqual(
            str(info.exception), "Value 1 is not a string", "Wrong error message"
        )
