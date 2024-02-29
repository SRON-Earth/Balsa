import pathlib

from ..data import load_dataset_bin, load_dataset_balsa
from ..util import run_program, get_statistics_from_time_file, \
                   get_classification_scores

class Driver:

    def __init__(self, path, data_format):

        self.path = pathlib.Path(path)
        self.data_format = data_format

    @staticmethod
    def add_default_config(config):

        config.add_classifier("balsa", driver="balsa", path="/path/to/dir/that/contains/balsa/binaries", data_format="bin")

    def get_data_format(self):

        return self.data_format

    def run(self, run_path, train_data_filename, train_label_filename, test_data_filename, test_label_filename, *,
            num_estimators, max_tree_depth, num_threads):

        run_statistics = {}

        args = ["-c", str(num_estimators), "-t", str(num_threads)]
        if max_tree_depth is not None:
            args += ["-d", str(max_tree_depth)]
        args += [str(train_data_filename), str(train_label_filename), "jigsaw.model"]
        result = run_program(self.path / "balsa_train", *args, log=True, time_file="train.time", cwd=run_path)
        get_statistics_from_time_file(run_path / "train.time", target_dict=run_statistics, key_prefix="train-")

        args = ["-t", str(num_threads), "-p", str(num_threads)]
        result = run_program(self.path / "balsa_classify", *args, "jigsaw.model", str(test_data_filename), "labels.bin", log=True, time_file="test.time", cwd=run_path)
        get_statistics_from_time_file(run_path / "test.time", target_dict=run_statistics, key_prefix="test-")

        for line in result.stdout.split("\n"):
            if "Maximum Depth" in line:
                run_statistics["train-max-tree-depth"] = int(line.split()[-1])
            if "Maximum Node Count" in line:
                run_statistics["train-max-node-count"] = int(line.split()[-1])
            if "Model Load Time:" in line:
                run_statistics["test-model-load-time"] = float(line.split()[-1])
            if "Data Load Time:" in line:
                run_statistics["test-data-load-time"] = float(line.split()[-1])
            if "Classification Time:" in line:
                run_statistics["test-classification-time"] = float(line.split()[-1])
            if "Label Store Time:" in line:
                run_statistics["test-label-store-time"] = float(line.split()[-1])

        labels = load_dataset_bin(test_label_filename) if self.data_format == "bin" else load_dataset_balsa(test_label_filename)
        predicted_labels = load_dataset_bin(run_path / "labels.bin") if self.data_format == "bin" else load_dataset_balsa(run_path / "labels.bin")
        get_classification_scores(predicted_labels, labels, target_dict=run_statistics, key_prefix="test-")

        return run_statistics
