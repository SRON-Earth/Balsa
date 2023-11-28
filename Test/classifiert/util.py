import numpy as np
import pathlib
import struct
import subprocess


def run_program(program, *args, log=False, time_file=None, timeout=None, cwd=None):

    program = pathlib.Path(program)
    if time_file is not None:
        result = subprocess.run(["time", "-v", "-o", time_file, str(program), *args], capture_output=True, text=True,
                                timeout=timeout, cwd=cwd)
    else:
        result = subprocess.run([str(program), *args], capture_output=True, text=True, timeout=timeout, cwd=cwd)
    if log:
        stdout_filename = f"{program.name}-stdout.txt"
        stdout_path = stdout_filename if cwd is None else cwd / stdout_filename
        with open(stdout_path, "w") as outputf:
            outputf.write(result.stdout)
        stderr_filename = f"{program.name}-stderr.txt"
        stderr_path = stderr_filename if cwd is None else cwd / stderr_filename
        with open(stderr_path, "w") as outputf:
            outputf.write(result.stderr)
    assert result.returncode == 0, f"Program '{program}' failed with exit code: {result.returncode}"
    return result


def get_statistics_from_time_file(time_file, *, target_dict=None, key_prefix=""):

    def rekey(key):
        return key_prefix + key

    if target_dict is None:
        target_dict = {}

    for line in pathlib.Path(time_file).read_text().split("\n"):
        if "User time" in line:
            target_dict[rekey("user-time")] = float(line.split()[-1])
        elif "System time" in line:
            target_dict[rekey("system-time")] = float(line.split()[-1])
        elif "Maximum resident set size" in line:
            target_dict[rekey("max-rss")] = int(line.split()[-1])

    return target_dict


def get_classification_scores(predicted_labels, labels, *, target_dict=None, key_prefix=""):

    predicted_labels, labels = np.asarray(predicted_labels), np.asarray(labels)

    num_true_positives  = np.sum((predicted_labels == 1.0) & (labels == 1.0))
    num_false_positives = np.sum((predicted_labels == 1.0) & (labels == 0.0))
    num_true_negatives  = np.sum((predicted_labels == 0.0) & (labels == 0.0))
    num_false_negatives = np.sum((predicted_labels == 0.0) & (labels == 1.0))

    num_total = num_true_positives + num_false_positives + num_true_negatives + num_false_negatives
    assert num_total == len(labels)

    def rekey(key):
        return key_prefix + key

    if target_dict is None:
        target_dict = {}

    target_dict[rekey("accuracy")] = (num_true_positives + num_true_negatives) / num_total
    target_dict[rekey("precision")] = num_true_positives / (num_true_positives + num_false_positives)
    target_dict[rekey("recall")] = num_true_positives / (num_true_positives + num_false_negatives)
    target_dict[rekey("P4-metric")] = (4.0 * num_true_positives * num_true_negatives) / ((4.0 * num_true_positives * num_true_negatives) + (num_true_positives + num_true_negatives) * (num_false_positives + num_false_negatives))

    return target_dict


def load_dataset_bin(filename):

    with open(filename, "rb") as inf:
        num_columns, = struct.unpack("<I", inf.read(4))
        unpacker = struct.Struct("<" + "f" * num_columns)
        if num_columns == 1:
            result = [row[0] for row in unpacker.iter_unpack(inf.read())]
        else:
            result = [list(row) for row in unpacker.iter_unpack(inf.read())]
    return result
