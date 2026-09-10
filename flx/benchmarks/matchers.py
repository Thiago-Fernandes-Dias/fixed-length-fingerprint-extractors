from abc import ABC, abstractmethod

import numpy as np

from flx.data.dataset import Identifier
from flx.data.embedding_loader import EmbeddingLoader

DEEPPRINT_MINIMUM_SCORE: float = -2.0
DEEPPRINT_MAXIMUM_SCORE: float = 2.0


class BiometricMatcher(ABC):
    @abstractmethod
    def similarity(self, sample1: Identifier, sample2: Identifier) -> float:
        raise NotImplementedError()


class VectorizedMatcher(BiometricMatcher):
    @abstractmethod
    def preload_vectorized(self, samples: list[Identifier]) -> None:
        """
        Preloads all samples into one numpy ndarray for vectorized comparison.
        """
        raise NotImplementedError()

    @abstractmethod
    def vectorized_similarity(self, sample: Identifier) -> np.ndarray[float]:
        """
        Similarities with all the samples in the preloaded vector.
        """
        raise NotImplementedError()


class CosineSimilarityMatcher(VectorizedMatcher):
    MINIMUM_SCORE: float = DEEPPRINT_MINIMUM_SCORE
    MAXIMUM_SCORE: float = DEEPPRINT_MAXIMUM_SCORE

    def __init__(self, embedding_dataset: EmbeddingLoader):
        self._embeddings = embedding_dataset
        self._matrix = None

    def similarity(self, sample1: Identifier, sample2: Identifier) -> float:
        emb1 = self._embeddings.get(sample1)
        emb2 = self._embeddings.get(sample2)
        raw_score = float(np.dot(emb1, emb2))
        normalized_score = (raw_score - self.MINIMUM_SCORE) / (
            self.MAXIMUM_SCORE - self.MINIMUM_SCORE
        )
        return float(np.clip(normalized_score, 0.0, 1.0))

    def preload_vectorized(self, samples: list[Identifier]) -> None:
        """
        Preloads all samples into one numpy ndarray for vectorized comparison.
        """
        vectors = [self._embeddings.get(s) for s in samples]
        self._matrix = np.stack(vectors)

    def vectorized_similarity(self, sample: Identifier) -> np.ndarray[float]:
        """
        Similarities for all the items in the preloaded vector.
        """
        emb = self._embeddings.get(sample)
        vector = emb.vector if hasattr(emb, "vector") else emb
        raw_vals = np.matmul(self._matrix, vector)
        normalized_vals = (raw_vals - self.MINIMUM_SCORE) / (
            self.MAXIMUM_SCORE - self.MINIMUM_SCORE
        )
        return np.clip(normalized_vals, 0.0, 1.0)

