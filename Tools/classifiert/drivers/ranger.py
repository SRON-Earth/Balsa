import csv
import numpy as np
import pathlib

from ..util import run_program, get_statistics_from_time_file, \
                   get_statistics_from_stdout, get_classification_scores

class Driver:

    def __init__(self, path):

        self.path = pathlib.Path(path)

    @staticmethod
    def add_default_config(config):

        config.add_classifier("ranger", driver="ranger", path="/path/to/dir/that/contains/ranger/binary")

    def get_data_format(self):

        return "csv"

    def run(self, run_path, train_data_filename, train_label_filename, test_data_filename, test_label_filename, *,
            num_estimators, random_seed, max_tree_depth, num_threads):

        assert train_label_filename is None
        assert test_label_filename is None

        run_statistics = {}

        args = ["--file", str(train_data_filename),
                "--depvarname", "label",
                "--treetype", "1",
                "--ntree", str(num_estimators),
                "--skipoob",
                "--noreplace",
                "--fraction", "1",
                "--outprefix", "ranger",
                "--write",
                "--nthreads", str(num_threads)]
        if random_seed is not None:
            args += ["--seed", str(random_seed)]
        if max_tree_depth is not None:
            args += ["--maxdepth", str(max_tree_depth)]
        result = run_program(self.path / "ranger", *args, log=True, log_prefix="ranger-train", time_file="train.time", cwd=run_path)
        get_statistics_from_time_file(run_path / "train.time", target_dict=run_statistics, key_prefix="train-")

        args = ["--file", str(test_data_filename),
                "--depvarname", "label",
                "--predict", "ranger.forest",
                "--nthreads", str(num_threads)]
        if random_seed is not None:
            args += ["--seed", str(random_seed)]
        run_program(self.path / "ranger", *args, log=True, log_prefix="ranger-test", time_file="test.time", cwd=run_path)
        get_statistics_from_time_file(run_path / "test.time", target_dict=run_statistics, key_prefix="test-")

        with open(run_path / "ranger_out.prediction", "r") as infile:
            assert infile.readline().startswith("Predictions")
            predicted_labels = [int(line) for line in infile]

        with open(test_data_filename, "r") as infile:
            reader = csv.reader(infile)
            headers = next(reader, None)
            assert headers[-1] == "label"
            labels = [int(row[-1]) for row in reader]

        assert len(labels) == len(predicted_labels)

        get_classification_scores(predicted_labels, labels, target_dict=run_statistics, key_prefix="test-")

        return run_statistics
