import pathlib

from ..data import load_dataset_balsa
from ..util import run_program, get_statistics_from_time_file, \
                   get_statistics_from_stdout, get_classification_scores

class Driver:

    def __init__(self, path):

        self.path = pathlib.Path(path)

    @staticmethod
    def add_default_config(config):

        config.add_classifier("balsa", driver="balsa", path="/path/to/dir/that/contains/balsa/binaries")

    def get_data_format(self):

        return "balsa"

    def run(self, run_path, train_data_filename, train_label_filename, test_data_filename, test_label_filename, *,
            num_estimators, random_seed, max_tree_depth, num_features, num_threads):

        run_statistics = {}

        args = ["-c", str(num_estimators), "-t", str(num_threads)]
        if random_seed is not None:
            args += ["-s", str(random_seed)]
        if max_tree_depth is not None:
            args += ["-d", str(max_tree_depth)]
        if num_features is not None:
            args += ["-f", str(num_features)]
        args += [str(train_data_filename), str(train_label_filename), "jigsaw.model"]
        result = run_program(self.path / "balsa_train", *args, log=True, time_file="train.time", cwd=run_path)
        get_statistics_from_time_file(run_path / "train.time", target_dict=run_statistics, key_prefix="train-")
        get_statistics_from_stdout(result.stdout, target_dict=run_statistics, key_prefix="train-")

        args = ["-t", str(num_threads), "-p", str(num_threads)]
        result = run_program(self.path / "balsa_classify", *args, "jigsaw.model", str(test_data_filename), log=True, time_file="test.time", cwd=run_path)
        get_statistics_from_time_file(run_path / "test.time", target_dict=run_statistics, key_prefix="test-")
        get_statistics_from_stdout(result.stdout, target_dict=run_statistics, key_prefix="test-")

        balsa_label_filename = run_path / (test_data_filename.stem + "-predictions.balsa")
        labels, predicted_labels = load_dataset_balsa(test_label_filename), load_dataset_balsa(balsa_label_filename)

        get_classification_scores(predicted_labels, labels, target_dict=run_statistics, key_prefix="test-")

        return run_statistics
