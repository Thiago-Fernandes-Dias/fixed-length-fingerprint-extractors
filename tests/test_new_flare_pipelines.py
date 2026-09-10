import unittest
import numpy as np
import torch

from flx.data.dataset import Dataset, Identifier, IdentifierSet, DataLoader
from flx.data.embedding_loader import FLAREEmbeddingLoader
from flx.extractor.flare import (
    FLAREEnhanceFirstPipeline,
    FLAREAlignOnlyPipeline,
    FLAREFullPipeline,
)
from flx.models.flare.fdd import FDD
from flx.benchmarks.matchers import (
    FLAREMatcher,
    warp_flare_representation,
    calculate_flare_score,
)


class DummyDataLoader(DataLoader):
    def get(self, identifier: Identifier) -> torch.Tensor:
        torch.manual_seed(hash(identifier.impression) % 1000)
        return torch.rand(1, 300, 300)


class TestNewFLAREPipelines(unittest.TestCase):
    def setUp(self):
        self.desc_path = "./pretrained_models/flare/desc/desc_model.pth.tar"
        self.voting_pose_path = "./pretrained_models/flare/pose/VotingPose.pth"
        self.regression_pose_path = "./pretrained_models/flare/pose/RegressionPose.pth"
        self.priorenh_dir = "./pretrained_models/flare/enhancement/priorenh"
        self.unetenh_path = "./pretrained_models/flare/enhancement/unetenh/unetenh.pth"

    def test_warp_flare_representation(self):
        feat = np.random.randn(3072).astype(np.float32)
        mask = (np.random.rand(256) > 0.2).astype(np.float32)
        pose = np.array([256.0, 256.0, 0.0])

        wf, wm = warp_flare_representation(feat, mask, pose)
        self.assertEqual(wf.shape, (3072,))
        self.assertEqual(wm.shape, (256,))

    def test_enhance_first_pipeline_structure_and_extract(self):
        pipeline = FLAREEnhanceFirstPipeline(
            desc_model_path=self.desc_path,
            voting_pose_path=self.voting_pose_path,
            regression_pose_path=self.regression_pose_path,
            priorenh_dir=self.priorenh_dir,
            unetenh_path=self.unetenh_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
            batch_size=2,
        )
        self.assertIsInstance(pipeline.model, FDD)
        self.assertFalse(hasattr(pipeline, "extractor"))

        ids = IdentifierSet([Identifier(0, 0), Identifier(0, 1)])
        dataset = Dataset(DummyDataLoader(), ids)

        loader = pipeline.extract(dataset)
        self.assertEqual(loader.features.shape, (2, 4, 3072))
        self.assertEqual(loader.masks.shape, (2, 4, 256))
        self.assertFalse(loader.has_poses)

        matcher = FLAREMatcher(loader)
        score = matcher.similarity(Identifier(0, 0), Identifier(0, 1))
        self.assertTrue(0.0 <= score <= 1.0)

    def test_align_only_pipeline_structure_and_matching(self):
        pipeline = FLAREAlignOnlyPipeline(
            desc_model_path=self.desc_path,
            voting_pose_path=self.voting_pose_path,
            regression_pose_path=self.regression_pose_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
            batch_size=2,
        )
        self.assertIsInstance(pipeline.model, FDD)
        self.assertFalse(hasattr(pipeline, "extractor"))
        self.assertFalse(hasattr(pipeline, "enh_prior"))
        self.assertFalse(hasattr(pipeline, "enh_unet"))

        ids = IdentifierSet([Identifier(0, 0), Identifier(0, 1)])
        dataset = Dataset(DummyDataLoader(), ids)

        loader = pipeline.extract(dataset)
        self.assertEqual(loader.features.shape, (2, 2, 3072))
        self.assertEqual(loader.masks.shape, (2, 2, 256))
        self.assertFalse(loader.has_poses)

        matcher = FLAREMatcher(loader)
        self.assertFalse(matcher._align_features)

        score = matcher.similarity(Identifier(0, 0), Identifier(0, 1))
        self.assertTrue(0.0 <= score <= 1.0)

        class MockComparison:
            def __init__(self, s1, s2):
                self.sample1 = s1
                self.sample2 = s2

        comparisons = [
            MockComparison(Identifier(0, 0), Identifier(0, 1)),
            MockComparison(Identifier(0, 0), Identifier(0, 0)),
        ]
        matcher.preload_for_benchmark(comparisons)
        score_00 = matcher.similarity(Identifier(0, 0), Identifier(0, 0))
        score_01 = matcher.similarity(Identifier(0, 0), Identifier(0, 1))
        self.assertTrue(0.0 <= score_00 <= 1.0)
        self.assertTrue(0.0 <= score_01 <= 1.0)
        self.assertGreater(score_00, score_01)


if __name__ == "__main__":
    unittest.main()
