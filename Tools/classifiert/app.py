import argparse
import datetime
import numpy as np
import pathlib
import sys

from .        import CLASSIFIERS
from .        import runners
from .datagen import get_train_dataset_filenames, get_test_dataset_filenames, \
                     generate_train_datasets, ingest_test_dataset, \
                     load_dataset_json, sample_dataset, store_dataset_bin
from .report  import write_report

RUN_DIR = pathlib.Path("run")


def profile(train_data_filename, test_data_filename, classifiers, data_sizes,
            num_estimators, max_tree_depth, num_threads, timeout, use_cache):

    print("\033[35m" + f"Generating train datasets..." + "\033[0m")
    generate_train_datasets(train_data_filename, data_sizes, use_cache=use_cache)

    print("\033[35m" + f"Ingesting test dataset..." + "\033[0m")
    ingest_test_dataset(test_data_filename)

    run_path = RUN_DIR / datetime.datetime.now().isoformat()
    run_path.mkdir()

    statistics = {}
    for classifier in classifiers:
        print("\033[35m" + f"Running {classifier} using {num_threads} threads..." + "\033[0m")

        classifier_run_path = run_path / classifier
        classifier_run_path.mkdir()

        classifier_descriptor = CLASSIFIERS[classifier]
        data_format = classifier_descriptor["data_format"]
        runner = classifier_descriptor["runner"]

        classifier_statistics = []

        for data_size in data_sizes:
            print("\033[32m" + str(data_size) + "\033[0m")
            train_data_filename, train_label_filename = get_train_dataset_filenames(data_size, data_format)
            test_data_filename, test_label_filename = get_test_dataset_filenames(data_format)

            run_statistics = {"data_size": data_size}
            test_run_path = classifier_run_path / str(data_size)
            test_run_path.mkdir()
            try:
                run_statistics.update(runner(test_run_path, train_data_filename, train_label_filename, test_data_filename,
                                             test_label_filename, num_estimators=num_estimators,
                                             max_tree_depth=max_tree_depth, num_threads=num_threads))
            except Exception as exception:
                print("\033[31m" + "Run failed: '" + str(exception) + "'." + "\033[0m")
            else:
                classifier_statistics.append(run_statistics)

        write_report(run_path / f"{classifier}.pdf", num_threads, {classifier: classifier_statistics})
        statistics[classifier] = classifier_statistics

    write_report(run_path / f"all.pdf", num_threads, statistics)


def sample(data_input_filename, data_output_filename, label_output_filename,
           sample_size, with_replacement, random_seed):

    data_points, labels = load_dataset_json(data_input_filename)
    assert len(data_points) == len(labels)

    if sample_size is None:
        new_data_points, new_labels = data_points, labels
    else:
        random_generator = np.random.default_rng(random_seed)
        new_data_points, new_labels = sample_dataset(data_points, labels, sample_size, random_generator=random_generator, replace=with_replacement)

    store_dataset_bin(data_output_filename, label_output_filename, new_data_points, new_labels)


def parse_command_line_arguments():

    def percentage(text):
        value = int(text)
        if value < 0 or value > 100:
            raise ValueError
        return value

    def positive_integer(text):
        value = int(text)
        if value <= 0:
            raise ValueError
        return value

    def data_size_list(text):
        return list(sorted([int(value) for value in text.split(",")]))

    parser = argparse.ArgumentParser(prog="classifiert", description="Tool to test random forest classifiers.")

    subparsers = parser.add_subparsers(dest="command")

    profile = subparsers.add_parser("profile", help="profile random forest classifiers")
    profile.add_argument("train_data_filename", metavar="TRAIN_DATA_FILE")
    profile.add_argument("test_data_filename", metavar="TEST_DATA_FILE")
    profile.add_argument("classifiers", metavar="CLASSIFIER", choices=CLASSIFIERS.keys(), nargs="+")
    profile.add_argument("-n", "--data-sizes", type=data_size_list, default="100,1000,2500,5000,10_000,25_000,50_000")
    profile.add_argument("-d", "--max-tree-depth", type=positive_integer, default="50")
    profile.add_argument("-e", "--num-estimators", type=positive_integer, default="150")
    profile.add_argument("-t", "--num-threads", type=positive_integer, default="1")
    profile.add_argument("-x", "--timeout", type=positive_integer, default=None)
    profile.add_argument("-C", "--no-cache", dest="use_cache", action="store_false")

    sample = subparsers.add_parser("sample", help="draw a sample from an existing (json-pickle) dataset")
    sample.add_argument("data_input_filename", metavar="DATA_INPUT_FILE")
    sample.add_argument("data_output_filename", metavar="DATA_OUTPUT_FILE")
    sample.add_argument("label_output_filename", metavar="LABEL_OUTPUT_FILE")
    sample.add_argument("-n", "--sample-size", type=positive_integer)
    sample.add_argument("-r", "--with-replacement", action="store_true")
    sample.add_argument("-s", "--random-seed", type=positive_integer)

    return parser.parse_args()


def main():

    args = parse_command_line_arguments()
    args_dict = dict(vars(args))

    command = args_dict.pop("command")
    if command == "profile":
        return profile(**args_dict)
    elif command == "sample":
        return sample(**args_dict)
    else:
        return -1
