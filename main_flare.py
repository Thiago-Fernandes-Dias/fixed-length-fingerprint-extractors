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
    subjects: list[int],
    gallery_impressions: list[int],
    query_impressions: list[int],
    extension: str = ".tif",
) -> object:
    image_loader = TransformedImageLoader(
        images=DirectoryImageLoader(db_path, extension=extension),
        transforms=[flare_image_transform],
    )

    test_dataset = Dataset(image_loader, image_loader.ids)
    embeddings = pipeline.extract(test_dataset)

    benchmark = create_verification_gallery_query_benchmark(
        subjects=subjects,
        gallery_impressions=gallery_impressions,
        query_impressions=query_impressions,
    )

    matcher = FLAREMatcher(embeddings)
    benchmark_result = benchmark.run(matcher)

    return benchmark_result


A_SUBJECTS = list(range(100))
B_SUBJECTS = list(range(101, 110))
GALLERY_IMPRESSIONS = list(range(0, 4))
QUERY_IMPRESSIONS = list(range(4, 8))

FOLDERS = [
    ("/mnt/d/Datasets/FVC/FVC2000/Dbs/Db1_a/tif", "../results/FLARE/fvc_2000_db1_a.csv", A_SUBJECTS),
    ("/mnt/d/Datasets/FVC/FVC2004/Dbs/Db1_a/tif", "../results/FLARE/fvc_2004_db1_a.csv", A_SUBJECTS),
]


def main():
    freeze_support()
    desc_model_path = "../FLARE/model_weights/desc_model.pth.tar"
    voting_pose_path = "../FLARE/model_weights/VotingPose.pth"
    regression_pose_path = "../FLARE/model_weights/RegressionPose.pth"
    priorenh_dir = "../FLARE_ENH/pretrained_model/priorenh"
    unetenh_path = "../FLARE_ENH/pretrained_model/unetenh/unetenh.pth"

    logging.info("Initializing Official FLARE Full Pipeline (2 Poses x 2 Enhancers = 4 Combinations)...")
    pipeline = FLAREFullPipeline(
        desc_model_path=desc_model_path,
        voting_pose_path=voting_pose_path,
        regression_pose_path=regression_pose_path,
        priorenh_dir=priorenh_dir,
        unetenh_path=unetenh_path,
        device="cuda",
    )

    for db_path, results_path, subjects in FOLDERS:
        if not os.path.exists(db_path):
            logging.info(f"Skipping non-existent dataset path: {db_path}")
            continue
        logging.info(f"Running Official FLARE 4-Combination Benchmark on: {db_path}")
        result = run_full_flare_benchmark(
            pipeline,
            db_path,
            subjects,
            GALLERY_IMPRESSIONS,
            QUERY_IMPRESSIONS,
        )
        result.save_scores(results_path)
        logging.info(f"Saved FLARE benchmark results to: {results_path}")


if __name__ == "__main__":
    freeze_support()
    main()
