import logging
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
from flx.scripts.generate_benchmarks import create_verification_benchmark
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO)


def run_benchmark(ex: DeepPrintExtractor, db_path: str, subjects: list[int], impressions_per_subject: int):
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

    benchmark = create_verification_benchmark(
        subjects=subjects,
        impressions_per_subject=list(range(impressions_per_subject))
    )

    matcher = CosineSimilarityMatcher(
        EmbeddingLoader.combine(tex_embeddings, minutia_embeddings)
    )

    results = benchmark.run(matcher)

    return results


def generate_graph(thresholds, fmr, fnmr, name):
    plt.figure()
    plt.plot(thresholds, fmr, label="FMR")
    plt.plot(thresholds, fnmr, label="FNMR")
    plt.xlabel("Limiar de corte")
    plt.ylabel("Taxa")
    plt.title("FMR e FNMR x Limiar de corte")
    plt.legend()
    plt.grid()
    plt.savefig(f"{name}.png")


def main():
    freeze_support()
    deep_print_tex_extractor: DeepPrintExtractor = get_DeepPrint_TexMinu(
        num_training_subjects=8000, num_dims=256
    )
    deep_print_tex_extractor.load_best_model(
        "./fixed-length-fingerprint-extractors/models"
    )
    base_path = "datasets/FVC/FVC2004/Dbs/"
    a_subjects = list(range(100))
    b_subjects = list(range(101, 110))
    impressions_per_subject = 8
    db1_a_result = run_benchmark(deep_print_tex_extractor, base_path + "DB1_A", a_subjects, impressions_per_subject)
    db1_b_result = run_benchmark(deep_print_tex_extractor, base_path + "DB1_B", b_subjects, impressions_per_subject)
    db2_a_result = run_benchmark(deep_print_tex_extractor, base_path + "DB2_A", a_subjects, impressions_per_subject)
    db2_b_result = run_benchmark(deep_print_tex_extractor, base_path + "DB2_B", b_subjects, impressions_per_subject)
    db3_a_result = run_benchmark(deep_print_tex_extractor, base_path + "DB3_A", a_subjects, impressions_per_subject)
    db3_b_result = run_benchmark(deep_print_tex_extractor, base_path + "DB3_B", b_subjects, impressions_per_subject)
    db4_a_result = run_benchmark(deep_print_tex_extractor, base_path + "DB4_A", a_subjects, impressions_per_subject)
    db4_b_result = run_benchmark(deep_print_tex_extractor, base_path + "DB4_B", b_subjects, impressions_per_subject)

    # Show ERR
    print(f"db1_a_results: EER: {db1_a_result.get_equal_error_rate()}")
    print(f"db1_b_results: EER: {db1_b_result.get_equal_error_rate()}")
    print(f"db2_a_results: EER: {db2_a_result.get_equal_error_rate()}")
    print(f"db2_b_results: EER: {db2_b_result.get_equal_error_rate()}")
    print(f"db3_a_results: EER: {db3_a_result.get_equal_error_rate()}")
    print(f"db3_b_results: EER: {db3_b_result.get_equal_error_rate()}")
    print(f"db4_a_results: EER: {db4_a_result.get_equal_error_rate()}")
    print(f"db4_b_results: EER: {db4_b_result.get_equal_error_rate()}")

    thresholds = [.5, .55, .6, .65, .7, .75, .8, .85, .9, .95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2.0]
    db1_a_fmrs = np.array(db1_a_result.false_match_rate(thresholds))
    db1_a_fnmrs = np.array(db1_a_result.false_non_match_rate(thresholds))
    db1_b_fmrs = np.array(db1_b_result.false_match_rate(thresholds))
    db1_b_fnmrs = np.array(db1_b_result.false_non_match_rate(thresholds))
    db2_a_fmrs = np.array(db2_a_result.false_match_rate(thresholds))
    db2_a_fnmrs = np.array(db2_a_result.false_non_match_rate(thresholds))
    db2_b_fmrs = np.array(db2_b_result.false_match_rate(thresholds))
    db2_b_fnmrs = np.array(db2_b_result.false_non_match_rate(thresholds))
    db3_a_fmrs = np.array(db3_a_result.false_match_rate(thresholds))
    db3_a_fnmrs = np.array(db3_a_result.false_non_match_rate(thresholds))
    db3_b_fmrs = np.array(db3_b_result.false_match_rate(thresholds))
    db3_b_fnmrs = np.array(db3_b_result.false_non_match_rate(thresholds))
    db4_a_fmrs = np.array(db4_a_result.false_match_rate(thresholds))
    db4_a_fnmrs = np.array(db4_a_result.false_non_match_rate(thresholds))
    db4_b_fmrs = np.array(db4_b_result.false_match_rate(thresholds))
    db4_b_fnmrs = np.array(db4_b_result.false_non_match_rate(thresholds))
    
    generate_graph(thresholds, db1_a_fmrs, db1_a_fnmrs, "DB1 A")
    generate_graph(thresholds, db1_b_fmrs, db1_b_fnmrs, "DB1 B")
    generate_graph(thresholds, db2_a_fmrs, db2_a_fnmrs, "DB2 A")
    generate_graph(thresholds, db2_b_fmrs, db2_b_fnmrs, "DB2 B")
    generate_graph(thresholds, db3_a_fmrs, db3_a_fnmrs, "DB3 A")
    generate_graph(thresholds, db3_b_fmrs, db3_b_fnmrs, "DB3 B")
    generate_graph(thresholds, db4_a_fmrs, db4_a_fnmrs, "DB4 A")
    generate_graph(thresholds, db4_b_fmrs, db4_b_fnmrs, "DB4 B")


if __name__ == '__main__':
    freeze_support()
    main()
