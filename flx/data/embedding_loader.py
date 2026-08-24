from typing import Union
from os.path import join
import json

import numpy as np

from flx.data.dataset import DataLoader, Identifier, IdentifierSet


class EmbeddingLoader(DataLoader):
    def __init__(self, identifiers: IdentifierSet, embeddings: np.ndarray):
        assert embeddings.ndim == 2
        assert len(identifiers) == embeddings.shape[0]
        self._id_to_idx = {id: idx for idx, id in enumerate(identifiers)}
        self._array = embeddings

    @property
    def ids(self) -> IdentifierSet:
        return IdentifierSet(self._id_to_idx.keys())

    @property
    def embedding_size(self) -> int:
        return self._array.shape[1]

    def get(self, id: Identifier) -> np.ndarray:
        return self._array[self._id_to_idx[id]]

    def numpy(self) -> np.ndarray:
        return self._array

    def save(self, outdir: str) -> None:
        """
        Saves embeddings and corresponding ids to disk

        @param outdir : absolute path of the output directory
        """
        outarr = self.numpy()
        outarr = outarr.astype(np.float32)
        np.save(join(outdir, "embeddings.npy"), outarr)
        with open(join(outdir, "ids.json"), "w") as file:
            json.dump(
                Identifier.ids_to_json(self._id_to_embedding.keys()), file, indent=None
            )

    @staticmethod
    def load(dir: str) -> "EmbeddingLoader":
        """
        Loads embedding dataset from disk

        @dir : absolute path of the embeddings dir

        @returns : Loaded embeddings dataset
        """
        with open(join(dir, "ids.json"), "r") as file:
            ids = Identifier.ids_from_json(json.load(file))
        array = np.load(join(dir, "embeddings.npy"))
        return EmbeddingLoader(ids, array)

    @staticmethod
    def combine(ds1: "EmbeddingLoader", ds2: "EmbeddingLoader") -> "EmbeddingLoader":
        if set(ds1.ids) != set(ds2.ids):
            raise ValueError(
                "To concatenate two EmbeddingLoaders they must contain the same ids!"
            )
        if ds1.embedding_size != ds2.embedding_size:
            raise ValueError(
                "To concatenate two EmbeddingLoaders they must have the same embedding size!"
            )
        return EmbeddingLoader(
            ds1.ids, np.concatenate([ds1.numpy(), ds2.numpy()], axis=1)
        )

    @staticmethod
    def combine_if_both_exist(
        ds1: "EmbeddingLoader", ds2: "EmbeddingLoader"
    ) -> "EmbeddingLoader":
        if ds1 is None:
            assert ds2 is not None
            return ds2
        if ds2 is None:
            assert ds1 is not None
            return ds1
        return EmbeddingLoader.combine(ds1, ds2)


class FLAREEmbeddingLoader(DataLoader):
    """
    DataLoader for FLARE descriptors: feature matrices and foreground masks per Identifier.
    Supports single representation per sample [N, D] or multi-combination representations [N, K, D].
    """
    def __init__(self, identifiers: IdentifierSet, features: np.ndarray, masks: np.ndarray):
        assert len(identifiers) == features.shape[0] == masks.shape[0]
        self._id_to_idx = {id: idx for idx, id in enumerate(identifiers)}
        self._features = features
        self._masks = masks

    @property
    def ids(self) -> IdentifierSet:
        return IdentifierSet(list(self._id_to_idx.keys()))

    def get(self, id: Identifier) -> tuple[np.ndarray, np.ndarray]:
        idx = self._id_to_idx[id]
        return self._features[idx], self._masks[idx]

    @property
    def features(self) -> np.ndarray:
        return self._features

    @property
    def masks(self) -> np.ndarray:
        return self._masks

    @property
    def is_multi_combination(self) -> bool:
        return self._features.ndim == 3
