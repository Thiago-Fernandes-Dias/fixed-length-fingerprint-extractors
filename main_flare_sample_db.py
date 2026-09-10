import logging
import os
from multiprocessing.dummy import freeze_support

from flx.extractor.flare import FLAREFullPipeline
from flx.data.image_loader import DirectoryImageLoader
from flx.data.transformed_image_loader import TransformedImageLoader
from flx.data.image_helpers import flare_image_transform
from flx.data.dataset import Dataset
from flx.scripts.generate_benchmarks import create_verification_gallery_query_benchmark
from flx.benchmarks.matchers import FLAREMatcher

logging.basicConfig(level=logging.INFO)


def run_full_flare_benchmark(
    pipeline: FLAREFullPipeline,
    db_path: str,
    gallery_impressions: list[int],
    query_impressions: list[int],
    subjects: list | None = None,
    extension: str = ".tif",
) -> object:
    image_loader = TransformedImageLoader(
        images=DirectoryImageLoader(db_path, extension=extension),
        transforms=[flare_image_transform],
    )

    if subjects is None:
        subjects = sorted(list({bid.subject for bid in image_loader.ids}))

    test_dataset = Dataset(image_loader, image_loader.ids)
    embeddings = pipeline.extract(test_dataset)

    benchmark = create_verification_gallery_query_benchmark(
        subjects=subjects,
        gallery_impressions=gallery_impressions,
        query_impressions=query_impressions,
    )

    matcher = FLAREMatcher(embeddings)
    return benchmark.run(matcher)


GALLERY_IMPRESSIONS = list(range(1, 5))
QUERY_IMPRESSIONS = list(range(5, 9))

DATASETS = [
    (
        "/media/thiago-dias/BACKUP/Datasets/CrossMatch_Sample_DB",
        "../results/FLARE/crossmatch_sample_db.parquet",
    ),
    (
        "/media/thiago-dias/BACKUP/Datasets/UareU_sample_DB",
        "../results/FLARE/uareu_sample_db.parquet",
    ),
]


def main():
    freeze_support()
    desc_model_path = "./pretrained_models/flare/desc/desc_model.pth.tar"
    voting_pose_path = "./pretrained_models/flare/pose/VotingPose.pth"
    regression_pose_path = "./pretrained_models/flare/pose/RegressionPose.pth"
    priorenh_dir = "./pretrained_models/flare/enhancement/priorenh"
    unetenh_path = "./pretrained_models/flare/enhancement/unetenh/unetenh.pth"

    logging.info(
        "Initializing Official FLARE Full Pipeline (2 Poses x 2 Enhancers = 4 Combinations)..."
    )
    pipeline = FLAREFullPipeline(
        desc_model_path=desc_model_path,
        voting_pose_path=voting_pose_path,
        regression_pose_path=regression_pose_path,
        priorenh_dir=priorenh_dir,
        unetenh_path=unetenh_path,
        device="cuda",
    )

    for db_path, results_path in DATASETS:
        if not os.path.exists(db_path):
            logging.info(f"Skipping non-existent dataset path: {db_path}")
            continue
        logging.info(f"Running Official FLARE 4-Combination Benchmark on: {db_path}")
        result = run_full_flare_benchmark(
            pipeline,
            db_path,
            GALLERY_IMPRESSIONS,
            QUERY_IMPRESSIONS,
        )
        result.save_scores(results_path)
        logging.info(f"Saved FLARE benchmark results to: {results_path}")


if __name__ == "__main__":
    freeze_support()
    main()
