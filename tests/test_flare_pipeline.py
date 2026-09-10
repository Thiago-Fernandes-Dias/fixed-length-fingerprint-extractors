import unittest
import torch
import numpy as np

from flx.extractor.flare import FLAREFullPipeline, FLAREExtractor
from flx.models.flare.fdd import FDD


class TestFLAREFullPipeline(unittest.TestCase):
    def test_pipeline_fdd_model_initialization(self):
        pipeline = FLAREFullPipeline(
            desc_model_path="",
            voting_pose_path="",
            regression_pose_path="",
            priorenh_dir="./pretrained_models/flare/enhancement/priorenh",
            unetenh_path="",
            device="cpu",
        )
        self.assertIsInstance(pipeline.model, FDD)
        self.assertFalse(hasattr(pipeline, "extractor"))

    def test_fdd_embedding_equivalence(self):
        pipeline = FLAREFullPipeline(
            desc_model_path="",
            voting_pose_path="",
            regression_pose_path="",
            priorenh_dir="./pretrained_models/flare/enhancement/priorenh",
            unetenh_path="",
            device="cpu",
        )
        extractor = FLAREExtractor(
            model_path="",
            device="cpu",
        )

        extractor.model.load_state_dict(pipeline.model.state_dict())

        pipeline.model.eval()
        extractor.model.eval()

        dummy_tensor = torch.randn(4, 1, 256, 256)
        with torch.no_grad():
            pipeline_out = pipeline.model.get_embedding(dummy_tensor)
            extractor_out = extractor.model.get_embedding(dummy_tensor)

        np.testing.assert_allclose(
            pipeline_out["feature"].numpy(),
            extractor_out["feature"].numpy(),
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            pipeline_out["mask"].numpy(),
            extractor_out["mask"].numpy(),
            rtol=1e-5,
            atol=1e-5,
        )


if __name__ == "__main__":
    unittest.main()
