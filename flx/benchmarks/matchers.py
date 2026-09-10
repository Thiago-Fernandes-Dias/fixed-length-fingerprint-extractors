from abc import ABC, abstractmethod

import numpy as np

from flx.data.dataset import Identifier
from flx.data.embedding_loader import EmbeddingLoader, FLAREEmbeddingLoader

DEEPPRINT_MINIMUM_SCORE = -2.0
DEEPPRINT_MAXIMUM_SCORE = 2.0
DEEPPRINT_MIN_SCORE = DEEPPRINT_MINIMUM_SCORE
DEEPPRINT_MAX_SCORE = DEEPPRINT_MAXIMUM_SCORE

FLARE_MINIMUM_SCORE = -1.0
FLARE_MAXIMUM_SCORE = 1.0
FLARE_MIN_SCORE = FLARE_MINIMUM_SCORE
FLARE_MAX_SCORE = FLARE_MAXIMUM_SCORE


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
    MINIMUM_SCORE = DEEPPRINT_MINIMUM_SCORE
    MAXIMUM_SCORE = DEEPPRINT_MAXIMUM_SCORE

    def __init__(self, embedding_dataset: EmbeddingLoader):
        self._embeddings = embedding_dataset
        self._matrix = None

    def similarity(self, sample1: Identifier, sample2: Identifier) -> float:
        emb1 = self._embeddings.get(sample1)
        emb2 = self._embeddings.get(sample2)
        raw_score = float(np.dot(emb1, emb2))
        normalized_score = (raw_score - DEEPPRINT_MINIMUM_SCORE) / (
            DEEPPRINT_MAXIMUM_SCORE - DEEPPRINT_MINIMUM_SCORE
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
        normalized_vals = (raw_vals - DEEPPRINT_MINIMUM_SCORE) / (
            DEEPPRINT_MAXIMUM_SCORE - DEEPPRINT_MINIMUM_SCORE
        )
        return np.clip(normalized_vals, 0.0, 1.0)


def calculate_flare_score(
    feat1: np.ndarray,
    feat2: np.ndarray,
    mask1: np.ndarray,
    mask2: np.ndarray,
    ndim_feat: int = 12,
    binary: bool = False,
) -> np.ndarray:
    if feat1.ndim == 1:
        feat1 = feat1[None, :]
    if feat2.ndim == 1:
        feat2 = feat2[None, :]
    if mask1.ndim == 1:
        mask1 = mask1[None, :]
    if mask2.ndim == 1:
        mask2 = mask2[None, :]

    feat1_dense = feat1
    feat1_mask = np.tile(mask1, (1, ndim_feat))
    feat2_dense = feat2
    feat2_mask = np.tile(mask2, (1, ndim_feat))

    if binary:
        feat1_dense = (feat1_dense > 0).astype(np.float32)
        feat2_dense = (feat2_dense > 0).astype(np.float32)
        feat1_mask = (feat1_mask > 0.5).astype(np.float32)
        feat2_mask = (feat2_mask > 0.2).astype(np.float32)

        n12 = np.matmul(feat1_mask, feat2_mask.T)
        d12 = (
            n12
            - np.matmul((feat1_mask * feat1_dense), (feat2_mask * feat2_dense).T)
            - np.matmul(
                (feat1_mask * (1 - feat1_dense)), (feat2_mask * (1 - feat2_dense)).T
            )
        )
        score = 1 - 2 * np.where(n12 > 0, d12 / np.clip(n12, 1e-3, None), 0.5)
    else:
        x1 = np.sqrt(np.matmul(feat1_mask * feat1_dense**2, feat2_mask.T))
        x2 = np.sqrt(np.matmul(feat1_mask, (feat2_dense**2 * feat2_mask).T))
        x12 = np.matmul(feat1_mask * feat1_dense, (feat2_mask * feat2_dense).T)
        score = x12 / np.clip(x1 * x2, 1e-3, None)

    return score


import cv2


def warp_flare_representation(
    feat: np.ndarray,
    mask: np.ndarray,
    pose: np.ndarray,
    feat_grid_shape: tuple[int, int] = (16, 16),
    middle_shape: tuple[int, int] = (512, 512),
) -> tuple[np.ndarray, np.ndarray]:
    f_grid = feat.reshape(12, 16, 16)
    m_grid = mask.reshape(1, 16, 16)

    scale = feat_grid_shape[0] / middle_shape[0]
    target_center = np.array(feat_grid_shape[::-1]) / 2.0

    img_c = pose[:2]
    theta = pose[2]

    theta_rad = np.deg2rad(theta)
    R = np.array([
        [np.cos(theta_rad), -np.sin(theta_rad)],
        [np.sin(theta_rad), np.cos(theta_rad)],
    ])
    trans = -img_c * scale
    t = np.dot(R, trans) + target_center
    M = np.array([[R[0, 0], R[0, 1], t[0]], [R[1, 0], R[1, 1], t[1]]], dtype=np.float32)

    warped_f = np.stack(
        [
            cv2.warpAffine(
                f,
                M,
                feat_grid_shape,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            for f in f_grid
        ],
        axis=0,
    )
    warped_m = cv2.warpAffine(
        m_grid[0],
        M,
        feat_grid_shape,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )[None]

    return warped_f.reshape(-1), warped_m.reshape(-1)


class FLAREMatcher(BiometricMatcher):
    """
    BiometricMatcher for FLARE descriptors (features & masks).
    Computes masked cosine similarity score as defined in the official FLARE model (IEEE TIFS 2026).
    Supports single representation or multi-combination representations taking max similarity score across combinations.
    When embeddings contain pose predictions, aligns representations to canonical pose before scoring.
    """
    MINIMUM_SCORE = FLARE_MINIMUM_SCORE
    MAXIMUM_SCORE = FLARE_MAXIMUM_SCORE

    def __init__(
        self,
        embeddings: FLAREEmbeddingLoader,
        ndim_feat: int = 12,
        binary: bool = False,
        align_features: bool | None = None,
    ):
        self._embeddings = embeddings
        self._ndim_feat = ndim_feat
        self._binary = binary
        self._align_features = (
            align_features if align_features is not None else embeddings.has_poses
        )
        self._cached_scores = {}

    def similarity(self, sample1: Identifier, sample2: Identifier) -> float:
        if (sample1, sample2) in self._cached_scores:
            return self._cached_scores[(sample1, sample2)]

        feat1, mask1 = self._embeddings.get(sample1)
        feat2, mask2 = self._embeddings.get(sample2)
        pose1 = self._embeddings.get_pose(sample1) if self._embeddings.has_poses else None
        pose2 = self._embeddings.get_pose(sample2) if self._embeddings.has_poses else None

        if feat1.ndim == 2 and feat2.ndim == 2:
            scores = []
            for k in range(feat1.shape[0]):
                f1_k, m1_k = feat1[k], mask1[k]
                f2_k, m2_k = feat2[k], mask2[k]
                if self._align_features and pose1 is not None and pose2 is not None:
                    f1_k, m1_k = warp_flare_representation(f1_k, m1_k, pose1[k])
                    f2_k, m2_k = warp_flare_representation(f2_k, m2_k, pose2[k])
                sc_mat = calculate_flare_score(
                    f1_k, f2_k, m1_k, m2_k, ndim_feat=self._ndim_feat, binary=self._binary
                )
                raw_score = float(sc_mat[0, 0])
                normalized_score = (raw_score - FLARE_MINIMUM_SCORE) / (
                    FLARE_MAXIMUM_SCORE - FLARE_MINIMUM_SCORE
                )
                scores.append(float(np.clip(normalized_score, 0.0, 1.0)))
            score = max(scores)
        else:
            f1, m1 = feat1, mask1
            f2, m2 = feat2, mask2
            if self._align_features and pose1 is not None and pose2 is not None:
                f1, m1 = warp_flare_representation(f1, m1, pose1)
                f2, m2 = warp_flare_representation(f2, m2, pose2)
            score_mat = calculate_flare_score(
                f1, f2, m1, m2, ndim_feat=self._ndim_feat, binary=self._binary
            )
            raw_score = float(score_mat[0, 0])
            normalized_score = (raw_score - FLARE_MINIMUM_SCORE) / (
                FLARE_MAXIMUM_SCORE - FLARE_MINIMUM_SCORE
            )
            score = float(np.clip(normalized_score, 0.0, 1.0))

        return score

    def preload_for_benchmark(self, comparisons: list) -> None:
        """
        Precomputes score matrix in batch for all comparisons in benchmark.
        For multi-combination representations, computes score matrix per combination
        and applies element-wise maximum across combinations (Eq. 8 in paper).
        """
        sample1_set = list(dict.fromkeys([c.sample1 for c in comparisons]))
        sample2_set = list(dict.fromkeys([c.sample2 for c in comparisons]))

        s1_indices = {s: i for i, s in enumerate(sample1_set)}
        s2_indices = {s: i for i, s in enumerate(sample2_set)}

        if self._embeddings.is_multi_combination:
            feats1 = np.stack([self._embeddings.get(s)[0] for s in sample1_set])
            masks1 = np.stack([self._embeddings.get(s)[1] for s in sample1_set])
            feats2 = np.stack([self._embeddings.get(s)[0] for s in sample2_set])
            masks2 = np.stack([self._embeddings.get(s)[1] for s in sample2_set])
            poses1 = (
                np.stack([self._embeddings.get_pose(s) for s in sample1_set])
                if self._embeddings.has_poses else None
            )
            poses2 = (
                np.stack([self._embeddings.get_pose(s) for s in sample2_set])
                if self._embeddings.has_poses else None
            )

            num_comb = feats1.shape[1]
            combination_matrices = []
            for k in range(num_comb):
                f1_k = feats1[:, k]
                m1_k = masks1[:, k]
                f2_k = feats2[:, k]
                m2_k = masks2[:, k]

                if self._align_features and poses1 is not None and poses2 is not None:
                    warped_1 = [warp_flare_representation(f1_k[i], m1_k[i], poses1[i, k]) for i in range(len(sample1_set))]
                    f1_k = np.array([w[0] for w in warped_1])
                    m1_k = np.array([w[1] for w in warped_1])
                    warped_2 = [warp_flare_representation(f2_k[j], m2_k[j], poses2[j, k]) for j in range(len(sample2_set))]
                    f2_k = np.array([w[0] for w in warped_2])
                    m2_k = np.array([w[1] for w in warped_2])

                sc_mat = calculate_flare_score(
                    f1_k, f2_k, m1_k, m2_k,
                    ndim_feat=self._ndim_feat, binary=self._binary
                )
                combination_matrices.append(sc_mat)

            score_matrix = np.maximum.reduce(combination_matrices)
        else:
            feats1 = np.stack([self._embeddings.get(s)[0] for s in sample1_set])
            masks1 = np.stack([self._embeddings.get(s)[1] for s in sample1_set])
            feats2 = np.stack([self._embeddings.get(s)[0] for s in sample2_set])
            masks2 = np.stack([self._embeddings.get(s)[1] for s in sample2_set])
            if self._align_features and self._embeddings.has_poses:
                poses1 = np.stack([self._embeddings.get_pose(s) for s in sample1_set])
                poses2 = np.stack([self._embeddings.get_pose(s) for s in sample2_set])
                warped_1 = [warp_flare_representation(feats1[i], masks1[i], poses1[i]) for i in range(len(sample1_set))]
                feats1 = np.array([w[0] for w in warped_1])
                masks1 = np.array([w[1] for w in warped_1])
                warped_2 = [warp_flare_representation(feats2[j], masks2[j], poses2[j]) for j in range(len(sample2_set))]
                feats2 = np.array([w[0] for w in warped_2])
                masks2 = np.array([w[1] for w in warped_2])

            score_matrix = calculate_flare_score(
                feats1, feats2, masks1, masks2, ndim_feat=self._ndim_feat, binary=self._binary
            )

        score_matrix = np.clip(
            (score_matrix - FLARE_MINIMUM_SCORE) / (FLARE_MAXIMUM_SCORE - FLARE_MINIMUM_SCORE),
            0.0,
            1.0,
        )

        for c in comparisons:
            i1 = s1_indices[c.sample1]
            i2 = s2_indices[c.sample2]
            self._cached_scores[(c.sample1, c.sample2)] = float(score_matrix[i1, i2])
