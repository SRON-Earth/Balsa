import pathlib

from .. import register_classifier
from ..util import run_program, get_statistics_from_time_file, get_classification_scores, load_dataset_bin

PACKAGE_DATA_PATH = pathlib.Path(__file__).parent.parent.absolute() / "data"


def run(run_path, train_data_filename, train_label_filename, test_data_filename, test_label_filename, *,
        num_estimators, max_tree_depth, num_threads):

    run_statistics = {}

    args = ["-e", str(num_estimators), "-t", str(num_threads)]
    if max_tree_depth is not None:
        args += ["-d", str(max_tree_depth)]
    args += [str(train_data_filename), str(train_label_filename), "sklearn.model"]

    result = run_program(PACKAGE_DATA_PATH / "sklearn-train.py", *args, log=True, time_file="train.time", cwd=run_path)
    get_statistics_from_time_file(run_path / "train.time", target_dict=run_statistics, key_prefix="train-")

    for line in result.stdout.split("\n"):
        if "max-tree-depth" in line:
            run_statistics["train-max-tree-depth"] = int(line.split()[-1])
        if "max-node-count" in line:
            run_statistics["train-max-node-count"] = int(line.split()[-1])

    result = run_program(PACKAGE_DATA_PATH / "sklearn-test.py",
                         "sklearn.model",
                         str(test_data_filename),
                         "labels.bin",
                         log=True,
                         time_file="test.time",
                         cwd=run_path)
    get_statistics_from_time_file(run_path / "test.time", target_dict=run_statistics, key_prefix="test-")

    labels = load_dataset_bin(test_label_filename)
    predicted_labels = load_dataset_bin(run_path / "labels.bin")
    get_classification_scores(predicted_labels, labels, target_dict=run_statistics, key_prefix="test-")

    return run_statistics

register_classifier("sklearn", "bin", run)
