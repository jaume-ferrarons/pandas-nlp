import unittest
import pandas as pd
import pandas.testing as pt

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
        cls._df_empty = pd.DataFrame({"id": [], "text": []})
        cls._df_language = pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "text": [
                    "I like cats",
                    "Me gustan los gatos",
                    "M'agraden els gats",
                    "J'aime les chats",
                    "Ich mag Katzen",
                ],
            }
        )

    def test_sentences(self):
        pt.assert_series_equal(
            self._df_sentences.text.nlp.sentences(),
            pd.Series(
                [[], ["Hello, how are you?"], ["Code.", "Sleep.", "Eat"]],
                name="text_sentences",
            ),
        )

    def test_sentences_empty(self):
        pt.assert_series_equal(
            self._df_empty.text.nlp.sentences(),
            pd.Series([], dtype="object", name="text_sentences"),
        )

    def test_embeddings(self):
        embeddings = self._df_words.text.nlp.embedding()
        self.assertEqual(len(embeddings), 3, "Incorrect number of embeddings returned")
        for embedding in embeddings:
            self.assertEqual(len(embedding), 96)

    def test_embeddings_empty(self):
        embeddings = self._df_empty.text.nlp.embedding()
        pt.assert_series_equal(
            embeddings,
            pd.Series([], dtype="object", name="text_embedding"),
        )

    def test_language(self):
        languages = self._df_language.text.nlp.language()
        pt.assert_series_equal(
            languages, pd.Series(["en", "es", "ca", "fr", "de"], name="text_language")
        )

    def test_language_empty(self):
        languages = self._df_empty.text.nlp.language()
        pt.assert_series_equal(
            languages,
            pd.Series([], dtype="object", name="text_language"),
        )

    def test_nlp_on_not_str(self):
        with self.assertRaises(TypeError) as info:
            self._df_words.id.nlp.sentences()
        self.assertEqual(
            str(info.exception), "Value 1 is not a string", "Wrong error message"
        )
