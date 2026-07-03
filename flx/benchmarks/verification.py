import os
import csv
import json

import tqdm
import numpy as np

from flx.benchmarks.matchers import BiometricMatcher
from flx.benchmarks.biometric_comparison import (
    BiometricComparison,
    BiometricGalleryComparison,
    BiometricComparisonResult,
    biometric_comparisons_to_json,
    biometric_comparisons_from_json,
    biometric_comparison_results_to_json,
    biometric_comparison_results_from_json,
)
from flx.data.dataset import Identifier


class VerificationResult:
    def __init__(self, comparison_results: list[BiometricComparisonResult]):
        self._mated_scores: list[float] = []
        self._mated_comparisons: list[BiometricComparisonResult] = []
        self._non_mated_scores: list[float] = []
        self._non_mated_comparisons: list[BiometricComparisonResult] = []
        for res in comparison_results:
            if res.comparison.mated:
                self._mated_scores.append(res.similarity)
                self._mated_comparisons.append(res)
                continue
            self._non_mated_scores.append(res.similarity)
            self._non_mated_comparisons.append(res)

        self._mated_scores: np.ndarray = np.array(self._mated_scores)
        self._non_mated_scores: np.ndarray = np.array(self._non_mated_scores)

    def threshold_for_fmr(self, fmr: float):
        """
        Returns the threshold corresponding to the given False Match Rate (FMR).
        FMR := #(non-mated comparisons that matched) / #(non-mated comparisons).
        """
        nms_sorted = np.sort(self._non_mated_scores)
        num_fm = int(self._non_mated_scores.shape[0] * fmr) + 1
        return nms_sorted[-num_fm]

    def false_match_rate(self, thresholds: list[float]) -> list[float]:
        """
        Returns the false match rate when the given thresholds are applied
        """
        ratios = []
        for t in thresholds:
            is_match = self._non_mated_scores >= t
            ratios.append(np.sum(is_match) / self._non_mated_scores.shape[0])
        return ratios

    def false_non_match_rate(self, thresholds: list[float]) -> list[float]:
        """
        Returns the false non match rate when the given thresholds are applied
        """
        ratios = []
        for t in thresholds:
            is_no_match = self._mated_scores < t
            ratios.append(np.sum(is_no_match) / self._mated_scores.shape[0])
        return ratios

    def get_mated_scores(self) -> np.ndarray[float]:
        return self._mated_scores

    def get_non_mated_scores(self) -> np.ndarray[float]:
        return self._non_mated_scores

    def get_equal_error_rate(self) -> float:
        scores = np.concatenate([self._mated_scores, self._non_mated_scores])
        is_mated = np.concatenate(
            [
                np.ones(len(self._mated_scores), dtype=bool),
                np.zeros(len(self._non_mated_scores), dtype=bool),
            ]
        )
        # Sort by similarity (ascending)
        idxs = np.argsort(scores)
        scores = scores[idxs]
        is_mated = is_mated[idxs]

        # Calculate number of mated comparisons in subset of comparisons
        num_mated_cumulative = np.cumsum(is_mated, dtype=np.uint32)
        num_non_mated_cumulative = np.cumsum(np.logical_not(is_mated), dtype=np.uint32)
        fnmr = num_mated_cumulative / len(self._mated_comparisons)
        fmr = (len(self._non_mated_comparisons) - num_non_mated_cumulative) / len(
            self._non_mated_comparisons
        )
        diff_errors = np.abs(fnmr - fmr)
        idx = np.argmin(diff_errors)
        return fmr[idx]

    def save(self, path: str) -> None:
        jsn = biometric_comparison_results_to_json(
            path, self._mated_comparisons + self._non_mated_comparisons
        )
        with open(path, "w") as f:
            json.dump(jsn, f)

    def to_csv(self, path: str) -> None:
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["user_1", "impression_1", "user_2", "impression_2", "score"])
            all_results = self._mated_comparisons + self._non_mated_comparisons
            for idx, res in enumerate(all_results):
                writer.writerow([
                    res.comparison.sample1.subject,
                    res.comparison.sample1.impression,
                    res.comparison.sample2.subject,
                    res.comparison.sample2.impression,
                    res.similarity,
                ])

    @staticmethod
    def load(path: str) -> "VerificationResult":
        if not os.path.exists(path):
            raise RuntimeError(f"Cannot load results: File {path} does not exist")
        with open(path, "r") as f:
            jsn = json.load(f)
        return VerificationResult(biometric_comparison_results_from_json(jsn))


class VerificationBenchmark:
    def __init__(self, comparisons: list[BiometricComparison]):
        self._comparisons: list[BiometricComparison] = comparisons

    def run(self, matcher: BiometricMatcher) -> VerificationResult:
        results = []
        for comp in tqdm.tqdm(self._comparisons):
            similarity = matcher.similarity(comp.sample1, comp.sample2)
            results.append(BiometricComparisonResult(comp, similarity))
        return VerificationResult(results)

    def save(self, path: str) -> None:
        jsn = biometric_comparisons_to_json(self._comparisons)
        with open(path, "w") as f:
            json.dump(jsn, f)

    @staticmethod
    def load(path: str) -> "VerificationBenchmark":
        with open(path, "r") as f:
            jsn = json.load(f)
        comparisons = biometric_comparisons_from_json(jsn)
        return VerificationBenchmark(comparisons)


class VerificationGalleryBenchmark:
    def __init__(
        self,
        gallery_comparisons: list[BiometricGalleryComparison],
        gallery_by_subject: dict[int, list[Identifier]],
    ):
        self._comparisons: list[BiometricGalleryComparison] = gallery_comparisons
        self._gallery: dict[int, list[Identifier]] = gallery_by_subject

    def run(self, matcher: BiometricMatcher) -> VerificationResult:
        results: list[BiometricComparisonResult] = []
        for comp in tqdm.tqdm(self._comparisons):
            gallery_samples = self._gallery[comp.gallery_subject]
            max_similarity = max(
                matcher.similarity(comp.query_sample, g) for g in gallery_samples
            )
            bc = BiometricComparison(
                comp.query_sample, gallery_samples[0]
            )
            results.append(BiometricComparisonResult(bc, max_similarity))
        return VerificationResult(results)
