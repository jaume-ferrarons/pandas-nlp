import unittest
import pandas as pd
import pandas.testing as pt

from tests.utils import register_pandas_nlp_if_not_available


class SeriesAccessorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        register_pandas_nlp_if_not_available()

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
        cls._df_themed = pd.DataFrame(
            {
                "id": [0, 1, 2, 3],
                "text": [
                    "My computer is broken",
                    "I went to a piano concert",
                    "Chocolate is my favourite",
                    "Mozart played the piano",
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
            self.assertEqual(len(embedding), 300)

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

    def test_language_confidence(self):
        languages = self._df_language.text.nlp.language(confidence=True)
        pt.assert_series_equal(
            languages,
            pd.Series(
                [
                    {"language": "en", "confidence": 0.8970903158187866},
                    {"language": "es", "confidence": 0.9820452928543091},
                    {"language": "ca", "confidence": 0.9998055100440979},
                    {"language": "fr", "confidence": 0.9997128844261169},
                    {"language": "de", "confidence": 0.9979948401451111},
                ],
                name="text_language",
            ),
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

    def test_closest(self):
        themes = self._df_themed.text.nlp.closest(["music", "informatics", "food"])
        pt.assert_series_equal(
            themes,
            pd.Series(["informatics", "music", "food", "music"], name="text_closest"),
        )

    def test_closest_empty(self):
        themes = self._df_empty.text.nlp.closest(["music", "informatics", "food"])
        pt.assert_series_equal(
            themes,
            pd.Series([], dtype="object", name="text_closest"),
        )
