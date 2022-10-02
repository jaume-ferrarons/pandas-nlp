from abc import ABC, abstractmethod
from typing import Tuple, List


class LanguageModel(ABC):
    @abstractmethod
    def language(self, text: str) -> Tuple[str, float]:
        """Returns the language with confidence of the text"""

    def languages(self, texts: List[str]) -> List[Tuple[str, float]]:
        return [self.language(t) for t in texts]
