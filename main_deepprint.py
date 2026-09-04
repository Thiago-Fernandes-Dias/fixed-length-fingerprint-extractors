import logging
import os
from multiprocessing.dummy import freeze_support

from flx.benchmarks.matchers import CosineSimilarityMatcher
from flx.data.dataset import Dataset
from flx.data.embedding_loader import EmbeddingLoader
from flx.data.image_helpers import pad_and_resize_to_deepprint_input_size
from flx.data.image_loader import FVC2004Loader
from flx.data.transformed_image_loader import TransformedImageLoader
from flx.extractor.fixed_length_extractor import (
    DeepPrintExtractor,
    get_DeepPrint_TexMinu,
)
from flx.image_processing.binarization import LazilyAllocatedBinarizer
from flx.scripts.generate_benchmarks import create_verification_gallery_query_benchmark
import matplotlib.pyplot as plt
from icecream import ic
import numpy as np

logging.basicConfig(level=logging.INFO)


def run_benchmark(ex: DeepPrintExtractor, db_path: str, subjects: list[int], gallery_impressions: list[int], query_impressions: list[int]):
    image_loader = TransformedImageLoader(
        images=FVC2004Loader(db_path),
        poses=None,
        transforms=[
            pad_and_resize_to_deepprint_input_size,
            LazilyAllocatedBinarizer(5.0),
        ],
    )

    test_dataset: Dataset = Dataset(image_loader, image_loader.ids)
    tex_embeddings, minutia_embeddings = ex.extract(test_dataset)

    benchmark = create_verification_gallery_query_benchmark(
        subjects=subjects,
        gallery_impressions=gallery_impressions,
        query_impressions=query_impressions,
    )

    matcher = CosineSimilarityMatcher(
        EmbeddingLoader.combine(tex_embeddings, minutia_embeddings)
    )

    benchmark_result = benchmark.run(matcher)

    return benchmark_result


A_SUBJECTS = list(range(100))
B_SUBJECTS = list(range(101, 110))
GALLERY_IMPRESSIONS = list(range(0, 4))
QUERY_IMPRESSIONS = list(range(4, 8))
FOLDERS = [
    ("/media/thiago-dias/BACKUP/Datasets/FVC/FVC2000/Dbs/Db1_a/tif", "../results/DeepPrint/fvc_2000_db1_a.csv", A_SUBJECTS),
    ("/media/thiago-dias/BACKUP/Datasets/FVC/FVC2000/Dbs/Db1_b/tif", "../results/DeepPrint/fvc_2000_db1_b.csv", B_SUBJECTS),
    ("/media/thiago-dias/BACKUP/Datasets/FVC/FVC2000/Dbs/Db2_a/tif", "../results/DeepPrint/fvc_2000_db2_a.csv", A_SUBJECTS),
    ("/media/thiago-dias/BACKUP/Datasets/FVC/FVC2000/Dbs/Db2_b/tif", "../results/DeepPrint/fvc_2000_db2_b.csv", B_SUBJECTS),
    ("/media/thiago-dias/BACKUP/Datasets/FVC/FVC2000/Dbs/Db3_a/tif", "../results/DeepPrint/fvc_2000_db3_a.csv", A_SUBJECTS),
    ("/media/thiago-dias/BACKUP/Datasets/FVC/FVC2000/Dbs/Db3_b/tif", "../results/DeepPrint/fvc_2000_db3_b.csv", B_SUBJECTS),
    ("/media/thiago-dias/BACKUP/Datasets/FVC/FVC2000/Dbs/Db4_a/tif", "../results/DeepPrint/fvc_2000_db4_a.csv", A_SUBJECTS),
    ("/media/thiago-dias/BACKUP/Datasets/FVC/FVC2000/Dbs/Db4_b/tif", "../results/DeepPrint/fvc_2000_db4_b.csv", B_SUBJECTS),
    ("/media/thiago-dias/BACKUP/Datasets/FVC/FVC2002/Dbs/Db1_a/tif", "../results/DeepPrint/fvc_2002_db1_a.csv", A_SUBJECTS),
    ("/media/thiago-dias/BACKUP/Datasets/FVC/FVC2002/Dbs/Db1_b/tif", "../results/DeepPrint/fvc_2002_db1_b.csv", B_SUBJECTS),
    ("/media/thiago-dias/BACKUP/Datasets/FVC/FVC2002/Dbs/Db2_a/tif", "../results/DeepPrint/fvc_2002_db2_a.csv", A_SUBJECTS),
    ("/media/thiago-dias/BACKUP/Datasets/FVC/FVC2002/Dbs/Db2_b/tif", "../results/DeepPrint/fvc_2002_db2_b.csv", B_SUBJECTS),
    ("/media/thiago-dias/BACKUP/Datasets/FVC/FVC2002/Dbs/Db3_a/tif", "../results/DeepPrint/fvc_2002_db3_a.csv", A_SUBJECTS),
    ("/media/thiago-dias/BACKUP/Datasets/FVC/FVC2002/Dbs/Db3_b/tif", "../results/DeepPrint/fvc_2002_db3_b.csv", B_SUBJECTS),
    ("/media/thiago-dias/BACKUP/Datasets/FVC/FVC2002/Dbs/Db4_a/tif", "../results/DeepPrint/fvc_2002_db4_a.csv", A_SUBJECTS),
    ("/media/thiago-dias/BACKUP/Datasets/FVC/FVC2002/Dbs/Db4_b/tif", "../results/DeepPrint/fvc_2002_db4_b.csv", B_SUBJECTS),
    ("/media/thiago-dias/BACKUP/Datasets/FVC/FVC2004/Dbs/Db1_a/tif", "../results/DeepPrint/fvc_2004_db1_a.csv", A_SUBJECTS),
    ("/media/thiago-dias/BACKUP/Datasets/FVC/FVC2004/Dbs/Db1_b/tif", "../results/DeepPrint/fvc_2004_db1_b.csv", B_SUBJECTS),
    ("/media/thiago-dias/BACKUP/Datasets/FVC/FVC2004/Dbs/Db2_a/tif", "../results/DeepPrint/fvc_2004_db2_a.csv", A_SUBJECTS),
    ("/media/thiago-dias/BACKUP/Datasets/FVC/FVC2004/Dbs/Db2_b/tif", "../results/DeepPrint/fvc_2004_db2_b.csv", B_SUBJECTS),
    ("/media/thiago-dias/BACKUP/Datasets/FVC/FVC2004/Dbs/Db3_a/tif", "../results/DeepPrint/fvc_2004_db3_a.csv", A_SUBJECTS),
    ("/media/thiago-dias/BACKUP/Datasets/FVC/FVC2004/Dbs/Db3_b/tif", "../results/DeepPrint/fvc_2004_db3_b.csv", B_SUBJECTS),
    ("/media/thiago-dias/BACKUP/Datasets/FVC/FVC2004/Dbs/Db4_a/tif", "../results/DeepPrint/fvc_2004_db4_a.csv", A_SUBJECTS),
    ("/media/thiago-dias/BACKUP/Datasets/FVC/FVC2004/Dbs/Db4_b/tif", "../results/DeepPrint/fvc_2004_db4_b.csv", B_SUBJECTS),
]


def main():
    freeze_support()
    deep_print_tex_extractor: DeepPrintExtractor = get_DeepPrint_TexMinu(
        num_training_subjects=8000, num_dims=256
    )
    deep_print_tex_extractor.load_model(
        "/media/thiago-dias/BACKUP/Datasets/models/deepprint_texminu_512.pyt"
    )

    for db_path, results_path, subjects in FOLDERS:
        result = run_benchmark(
            deep_print_tex_extractor, db_path, subjects,
            GALLERY_IMPRESSIONS, QUERY_IMPRESSIONS,
        )
        result.save_scores(results_path)
        logging.info(f"Saved DeepPrint benchmark results to: {results_path}")

if __name__ == '__main__':
    freeze_support()
    main()
