import csv
import numpy as np

from .. import register_classifier
from ..util import run_program, get_statistics_from_time_file, get_classification_scores


def load_dataset_bin(filename):

    import struct

    with open(filename, "rb") as inf:
        num_columns, = struct.unpack("<I", inf.read(4))
        unpacker = struct.Struct("<" + "f" * num_columns)
        if num_columns == 1:
            result = [row[0] for row in unpacker.iter_unpack(inf.read())]
        else:
            result = [list(row) for row in unpacker.iter_unpack(inf.read())]
    return result


def run(run_path, train_data_filename, train_label_filename, test_data_filename, test_label_filename, *,
        num_estimators, max_tree_depth, num_threads):

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

    if max_tree_depth is not None:
        args += ["--maxdepth", str(max_tree_depth)]

    result = run_program("ranger", *args, log=True, time_file="train.time", cwd=run_path)
    get_statistics_from_time_file(run_path / "train.time", target_dict=run_statistics, key_prefix="train-")

    run_program("ranger",
                "--file", str(test_data_filename),
                "--depvarname", "label",
                "--predict", "ranger.forest",
                "--nthreads", str(num_threads),
                log=True,
                time_file="test.time",
                cwd=run_path)
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

register_classifier("ranger", "csv", run)
