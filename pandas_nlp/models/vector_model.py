from abc import ABC, abstractmethod
from typing import List
import numpy as np


class VectorModel(ABC):
    @abstractmethod
    def text_vector(self, text: str) -> np.ndarray:
        """Returns the embedding of the provided text"""

    def texts_vectors(self, texts: List[str]) -> List[np.ndarray]:
        return [self.text_vector(t) for t in texts]

    def closest(self, texts: List[str], labels: List[str]) -> List[str]:
        texts_embeddings = np.vstack(self.texts_vectors(texts))
        labels_embeddings = np.vstack(self.texts_vectors(labels))
        texts_embeddings = (
            texts_embeddings / np.sqrt((texts_embeddings**2).sum(-1))[..., np.newaxis]
        )
        labels_embeddings = (
            labels_embeddings
            / np.sqrt((labels_embeddings**2).sum(-1))[..., np.newaxis]
        )
        closest = np.argmax(np.dot(texts_embeddings, labels_embeddings.T), axis=1)
        return np.array(labels)[closest]
