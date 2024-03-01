import pathlib

from ..data import load_dataset_bin
from ..util import run_program, get_statistics_from_time_file, \
                   get_statistics_from_stdout, get_classification_scores

PACKAGE_DATA_PATH = pathlib.Path(__file__).parent.parent.absolute() / "scripts"

class Driver:

    def __init__(self, python):

        self.python = pathlib.Path(python)

    @staticmethod
    def add_default_config(config):

        config.add_classifier("sklearn", driver="sklearn", python="/path/to/python/interpreter")

    def get_data_format(self):

        return "bin"

    def run(self, run_path, train_data_filename, train_label_filename, test_data_filename, test_label_filename, *,
            num_estimators, max_tree_depth, num_threads):

        train_script = PACKAGE_DATA_PATH / "sklearn-train.py"
        test_script = PACKAGE_DATA_PATH / "sklearn-test.py"

        run_statistics = {}

        args = [str(train_script), "-e", str(num_estimators), "-t", str(num_threads)]
        if max_tree_depth is not None:
            args += ["-d", str(max_tree_depth)]
        args += [str(train_data_filename), str(train_label_filename), "sklearn.model"]

        result = run_program(self.python, *args, log=True, log_prefix="sklearn-train", time_file="train.time", cwd=run_path)
        get_statistics_from_time_file(run_path / "train.time", target_dict=run_statistics, key_prefix="train-")
        get_statistics_from_stdout(result.stdout, target_dict=run_statistics, key_prefix="train-")

        result = run_program(self.python,
                             str(test_script),
                             "sklearn.model",
                             str(test_data_filename),
                             "labels.bin",
                             log=True,
                             log_prefix="sklearn-test",
                             time_file="test.time",
                             cwd=run_path)
        get_statistics_from_time_file(run_path / "test.time", target_dict=run_statistics, key_prefix="test-")
        get_statistics_from_stdout(result.stdout, target_dict=run_statistics, key_prefix="test-")

        labels = load_dataset_bin(test_label_filename)
        predicted_labels = load_dataset_bin(run_path / "labels.bin")
        get_classification_scores(predicted_labels, labels, target_dict=run_statistics, key_prefix="test-")

        return run_statistics
