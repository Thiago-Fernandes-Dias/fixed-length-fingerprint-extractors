import numpy as np

from flx.benchmarks.matchers import CosineSimilarityMatcher
from flx.data.dataset import Identifier, IdentifierSet
from flx.data.embedding_loader import EmbeddingLoader


def test_cosine_similarity_matcher_normalization():
    ids = [Identifier(1, 0), Identifier(1, 1), Identifier(2, 0), Identifier(2, 1)]
    embeddings = np.array([
        [1.0, 1.0],
        [1.0, 1.0],
        [1.0, -1.0],
        [-1.0, -1.0],
    ])
    loader = EmbeddingLoader(IdentifierSet(ids), embeddings)
    matcher = CosineSimilarityMatcher(loader)

    score_max = matcher.similarity(Identifier(1, 0), Identifier(1, 1))
    assert np.isclose(score_max, 1.0)

    score_mid = matcher.similarity(Identifier(1, 0), Identifier(2, 0))
    assert np.isclose(score_mid, 0.5)

    score_min = matcher.similarity(Identifier(1, 0), Identifier(2, 1))
    assert np.isclose(score_min, 0.0)


def test_cosine_similarity_matcher_vectorized():
    gallery_ids = [Identifier(1, 1), Identifier(2, 0), Identifier(2, 1)]
    all_ids = [Identifier(1, 0)] + gallery_ids
    embeddings = np.array([
        [1.0, 1.0],
        [1.0, 1.0],
        [1.0, -1.0],
        [-1.0, -1.0],
    ])
    loader = EmbeddingLoader(IdentifierSet(all_ids), embeddings)
    matcher = CosineSimilarityMatcher(loader)

    matcher.preload_vectorized(gallery_ids)
    scores = matcher.vectorized_similarity(Identifier(1, 0))

    assert scores.shape == (3,)
    assert np.isclose(scores[0], 1.0)
    assert np.isclose(scores[1], 0.5)
    assert np.isclose(scores[2], 0.0)
