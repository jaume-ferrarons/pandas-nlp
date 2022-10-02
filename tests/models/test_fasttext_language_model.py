import unittest

from pandas_nlp.models.fasttext_language_model import FastTextLanguageModel


class FastTextLanguageModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._texts = [
            "I like cats",
            "Me gustan los gatos",
            "M'agraden els gats",
            "J'aime les chats",
            "Ich mag Katzen",
        ]

    def test_language(self):
        model = FastTextLanguageModel()
        language, score = model.language("J'aime les chats")
        self.assertEqual(language, "fr")
        self.assertAlmostEqual(score, 0.999, places=2)

    def test_languages(self):
        model = FastTextLanguageModel()
        languages = [t[0] for t in model.languages(self._texts)]
        self.assertListEqual(languages, ["en", "es", "ca", "fr", "de"])
