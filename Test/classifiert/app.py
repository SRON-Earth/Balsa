import argparse
import datetime
import pathlib
import sys

from .        import CLASSIFIERS
from .        import runners
from .datagen import get_dataset_filenames, generate_datasets
from .report  import write_report

RUN_DIR = pathlib.Path("run")


def app_main(classifiers, data_sizes, test_percentage, num_estimators, max_tree_depth, num_threads, timeout, use_cache):

    print("\033[35m" + f"Generating datasets..." + "\033[0m")
    generate_datasets(data_sizes, test_percentage, use_cache=use_cache)

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
            train_data_filename, train_label_filename, test_data_filename, test_label_filename = \
                get_dataset_filenames(data_size, test_percentage, data_format)
            test_size = 2 * round(data_size // 2 * test_percentage / 100)
            training_size = data_size - test_size
            run_statistics = {"data_size": data_size, "train_data_size": training_size, "test_data_size": test_size}
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
    parser.add_argument("classifiers", metavar="CLASSIFIER", choices=CLASSIFIERS.keys(), nargs="+")
    parser.add_argument("-n", "--data-sizes", type=data_size_list, default="100,1000,2500,5000,10_000,25_000,50_000")
    parser.add_argument("-p", "--test-percentage", type=percentage, default="20")
    parser.add_argument("-d", "--max-tree-depth", type=positive_integer, default="50")
    parser.add_argument("-e", "--num-estimators", type=positive_integer, default="150")
    parser.add_argument("-t", "--num-threads", type=positive_integer, default="1")
    parser.add_argument("-x", "--timeout", type=positive_integer, default=None)
    parser.add_argument("-C", "--no-cache", dest="use_cache", action="store_false")
    return parser.parse_args()


def main():

    args = parse_command_line_arguments()
    app_main(**dict(vars(args)))
