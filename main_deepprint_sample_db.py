import logging
import os
from multiprocessing.dummy import freeze_support

from flx.benchmarks.matchers import CosineSimilarityMatcher
from flx.data.dataset import Dataset
from flx.data.embedding_loader import EmbeddingLoader
from flx.data.image_helpers import pad_and_resize_to_deepprint_input_size
from flx.data.image_loader import DirectoryImageLoader
from flx.data.transformed_image_loader import TransformedImageLoader
from flx.extractor.fixed_length_extractor import (
    DeepPrintExtractor,
    get_DeepPrint_TexMinu,
)
from flx.image_processing.binarization import LazilyAllocatedBinarizer
from flx.scripts.generate_benchmarks import create_verification_gallery_query_benchmark

logging.basicConfig(level=logging.INFO)


def run_benchmark(
    extractor: DeepPrintExtractor,
    db_path: str,
    gallery_impressions: list[int],
    query_impressions: list[int],
    subjects: list | None = None,
    extension: str = ".tif",
):
    image_loader = TransformedImageLoader(
        images=DirectoryImageLoader(db_path, extension=extension),
        poses=None,
        transforms=[
            pad_and_resize_to_deepprint_input_size,
            LazilyAllocatedBinarizer(5.0),
        ],
    )

    if subjects is None:
        subjects = sorted(list({bid.subject for bid in image_loader.ids}))

    test_dataset = Dataset(image_loader, image_loader.ids)
    texture_embeddings, minutia_embeddings = extractor.extract(test_dataset)

    benchmark = create_verification_gallery_query_benchmark(
        subjects=subjects,
        gallery_impressions=gallery_impressions,
        query_impressions=query_impressions,
    )

    matcher = CosineSimilarityMatcher(
        EmbeddingLoader.combine(texture_embeddings, minutia_embeddings)
    )

    return benchmark.run(matcher)


GALLERY_IMPRESSIONS = list(range(1, 5))
QUERY_IMPRESSIONS = list(range(5, 9))

DATASETS = [
    (
        "/media/thiago-dias/BACKUP/Datasets/CrossMatch_Sample_DB",
        "../results/DeepPrint/crossmatch_sample_db.parquet",
    ),
    (
        "/media/thiago-dias/BACKUP/Datasets/UareU_sample_DB",
        "../results/DeepPrint/uareu_sample_db.parquet",
    ),
]


def main():
    freeze_support()
    deep_print_extractor = get_DeepPrint_TexMinu(
        num_training_subjects=8000, num_dims=256
    )
    deep_print_extractor.load_model(
        "./pretrained_models/deepprint/deepprint_texminu_512.pyt"
    )

    for db_path, results_path in DATASETS:
        if not os.path.exists(db_path):
            logging.info(f"Skipping non-existent dataset path: {db_path}")
            continue
        logging.info(f"Running DeepPrint benchmark on: {db_path}")
        result = run_benchmark(
            deep_print_extractor,
            db_path,
            GALLERY_IMPRESSIONS,
            QUERY_IMPRESSIONS,
        )
        result.save_scores(results_path)
        logging.info(f"Saved DeepPrint benchmark results to: {results_path}")


if __name__ == "__main__":
    freeze_support()
    main()
