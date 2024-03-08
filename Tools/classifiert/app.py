import argparse
import datetime
import numpy as np
import pathlib
import sys

from .config  import Configuration, load_config, store_config
from .drivers import get_drivers, get_driver
from .data    import set_cache_dir, get_train_dataset_filenames, \
                     get_test_dataset_filenames, generate_datasets, \
                     ingest_test_dataset, load_labelled_dataset_json, \
                     sample_dataset, store_labelled_dataset
from .report  import write_report

def generate_default_config_file(filename):

    config = Configuration()
    for driver_name in get_drivers():
        driver = get_driver(driver_name)
        driver.add_default_config(config)
    store_config(filename, config)

def profile(train_data_filename, test_data_filename, classifiers, config_file, data_sizes, test_percentage, num_estimators,
            random_seed, max_tree_depth, num_features, num_threads, timeout, use_cache):

    # Load configuration. If the configuration file does not exist, generate a
    # default configuration file.
    try:
        config = load_config(config_file)
    except FileNotFoundError:
        print("\033[35m" + f"Generating default configuration file..." + "\033[0m")
        generate_default_config_file(config_file)
        print(f"Generated configuration file: '{config_file}'.")
        print("Please edit the generated configuration file and try again.")
        return

    # Set cache directory based on configuration.
    set_cache_dir(config.cache_dir)

    # Generate datasets.
    print("\033[35m" + f"Generating datasets..." + "\033[0m")
    if test_data_filename is not None and test_percentage is not None:
        raise Exception("Either a test percentage or a test data file should be specified.")
    if test_data_filename is None and test_percentage is None:
        test_percentage = 33
    generate_datasets(train_data_filename, data_sizes, use_cache=use_cache, test_percentage=test_percentage, seed=random_seed)

    # Optionally ingest the specified test dataset.
    if test_data_filename is not None:
        print("\033[35m" + f"Ingesting test dataset..." + "\033[0m")
        ingest_test_dataset(test_data_filename)

    # Create run path.
    run_path = config.run_dir / datetime.datetime.now().isoformat()
    run_path.mkdir()

    # Run all classifiers.
    statistics = {}
    for classifier_name in classifiers:

        print("\033[35m" + f"Running {classifier_name} using {num_threads} threads..." + "\033[0m")

        try:
            classifier = config.get_classifier(classifier_name)
        except KeyError:
            print("\033[31m" + "Unknown classifier: '" + classifier_name + "'." + "\033[0m")
            continue

        driver = get_driver(classifier["driver"])(**classifier["args"])
        data_format = driver.get_data_format()

        classifier_run_path = run_path / classifier_name
        classifier_run_path.mkdir()

        classifier_statistics = []
        for data_size in data_sizes:
            print("\033[32m" + str(data_size) + "\033[0m")
            run_train_data_filename, run_train_label_filename = get_train_dataset_filenames(data_format, data_size, test_percentage)
            if test_data_filename is not None:
                run_test_data_filename, run_test_label_filename = get_test_dataset_filenames(data_format)
            else:
                run_test_data_filename, run_test_label_filename = get_test_dataset_filenames(data_format, data_size, test_percentage)

            run_statistics = {"data_size": data_size}
            test_run_path = classifier_run_path / str(data_size)
            test_run_path.mkdir()
            try:
                run_statistics.update(driver.run(test_run_path,
                                                 run_train_data_filename,
                                                 run_train_label_filename,
                                                 run_test_data_filename,
                                                 run_test_label_filename,
                                                 num_estimators=num_estimators,
                                                 random_seed=random_seed,
                                                 max_tree_depth=max_tree_depth,
                                                 num_features=num_features,
                                                 num_threads=num_threads))
            except Exception as exception:
                print("\033[31m" + "Run failed: '" + str(exception) + "'." + "\033[0m")
            else:
                classifier_statistics.append(run_statistics)

        write_report(run_path / f"{classifier_name}.pdf", num_threads, {classifier_name: classifier_statistics})
        statistics[classifier_name] = classifier_statistics

    # Write combined report.
    write_report(run_path / f"all.pdf", num_threads, statistics)

def sample(data_input_filename, data_output_filename, label_output_filename,
           data_format, sample_size, with_replacement, random_seed):

    data_points, labels = load_labelled_dataset_json(data_input_filename)
    assert len(data_points) == len(labels)

    if sample_size is None:
        new_data_points, new_labels = data_points, labels
    else:
        random_generator = np.random.default_rng(random_seed)
        new_data_points, new_labels = sample_dataset(data_points, labels, sample_size, random_generator=random_generator, replace=with_replacement)

    store_labelled_dataset(data_format, data_output_filename, label_output_filename, new_data_points, new_labels)

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
    profile.add_argument("train_data_filename", type=pathlib.Path, metavar="TRAIN_DATA_FILE")
    profile.add_argument("classifiers", metavar="CLASSIFIER", nargs="+")
    profile.add_argument("-c", "--config-file", type=pathlib.Path, default="classifiert.ini")
    profile.add_argument("-n", "--data-sizes", type=data_size_list)
    profile.add_argument("-p", "--test-percentage", type=percentage)
    profile.add_argument("-T", "--test-data-file", type=pathlib.Path, metavar="TEST_DATA_FILE", dest="test_data_filename")
    profile.add_argument("-d", "--max-tree-depth", type=positive_integer, default="50")
    profile.add_argument("-f", "--num-features", type=positive_integer)
    profile.add_argument("-e", "--num-estimators", type=positive_integer, default="150")
    profile.add_argument("-t", "--num-threads", type=positive_integer, default="1")
    profile.add_argument("-x", "--timeout", type=positive_integer, default=None)
    profile.add_argument("-s", "--random-seed", type=positive_integer)
    profile.add_argument("-C", "--no-cache", dest="use_cache", action="store_false")

    sample = subparsers.add_parser("sample", help="draw a sample from an existing (json-pickle) dataset")
    sample.add_argument("data_input_filename", type=pathlib.Path, metavar="DATA_INPUT_FILE")
    sample.add_argument("data_output_filename", type=pathlib.Path, metavar="DATA_OUTPUT_FILE")
    sample.add_argument("label_output_filename", type=pathlib.Path, nargs="?", metavar="LABEL_OUTPUT_FILE")
    sample.add_argument("-f", "--data-format", choices=("csv", "bin", "balsa"), default="bin")
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
