from abc import ABC, abstractmethod
from typing import List


class SentenceModel(ABC):
    @abstractmethod
    def text_sentences(self, text: str) -> List[str]:
        """Returns the sentences from the text"""

    def texts_sentences(self, texts: List[str]) -> List[List[str]]:
        return [self.text_sentences(t) for t in texts]
