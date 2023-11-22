import numpy as np

from .. import register_classifier
from ..util import run_program, get_statistics_from_time_file


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

    TIME_FILE = "time.txt"

    args = ["-c", str(num_estimators), "-t", str(num_threads)]
    if max_tree_depth is not None:
        args += ["-d", str(max_tree_depth)]
    args += [str(train_data_filename), str(train_label_filename), "jigsaw.model"]

    run_program("rftrain", *args, time_file=TIME_FILE, cwd=run_path)

    run_statistics = {}
    run_statistics.update(get_statistics_from_time_file(run_path / TIME_FILE))

    result = run_program("rfclassify", "jigsaw.model", str(test_data_filename), "labels.bin", cwd=run_path)

    for line in result.stdout.split("\n"):
        if "Maximum Depth" in line:
            run_statistics["depth"] = int(line.split()[-1])
        if "Maximum Node Count" in line:
            run_statistics["node_count"] = int(line.split()[-1])

    labels = load_dataset_bin(test_label_filename)
    predicted_labels = load_dataset_bin(run_path / "labels.bin")
    assert len(labels) == len(predicted_labels)

    accuracy = 1.0 - np.sum(np.asarray(predicted_labels) != np.asarray(labels)) / len(labels)
    run_statistics["accuracy"] = accuracy

    return run_statistics

register_classifier("jigsaw", "bin", run)
