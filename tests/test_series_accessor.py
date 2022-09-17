import unittest
import pandas as pd

import pandas_nlp


class SeriesAccessorTest(unittest.TestCase):
    def test_sentences(self):
        df = pd.DataFrame(
            {"id": [1, 2, 3], "text": ["", "Hello, how are you?", "Code. Sleep. Eat"]}
        )
        self.assertListEqual(
            df.text.nlp.sentences(),
            [[], ["Hello, how are you?"], ["Code.", "Sleep.", "Eat"]],
        )

    def test_embeddings(self):
        df = pd.DataFrame({"id": [1, 2, 3], "text": ["cat", "dog", "violin"]})
        embeddings = df.text.nlp.embeddings()
        self.assertEqual(len(embeddings), 3, "Incorrect number of embeddings returned")
        for embedding in embeddings:
            self.assertEqual(len(embedding), 96)
