import numpy as np
import pathlib
import re
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

def parse_elapsed_time(text):

    match = re.match(r"(?:([0-9]+)\:)?([0-9]+)\:([0-9]+(?:\.[0-9]+)?)", text)
    assert match is None or len(match.groups()) == 3

    if match is None:
        return None

    elapsed_time = 0.0
    for value, weight in zip(match.groups(), (3600.0, 60.0, 1.0)):
        if value is not None:
            elapsed_time += float(value) * weight
    return elapsed_time

def get_statistics_from_time_file(time_file, *, target_dict=None, key_prefix=""):

    if target_dict is None:
        target_dict = {}

    for line in pathlib.Path(time_file).read_text().split("\n"):
        if "User time" in line:
            target_dict[key_prefix + "user-time"] = float(line.split()[-1])
        elif "System time" in line:
            target_dict[key_prefix + "system-time"] = float(line.split()[-1])
        elif "Percent of CPU" in line:
            text = line.split()[-1]
            assert text[-1] == "%"
            target_dict[key_prefix + "percent-cpu"] = float(text[:-1])
        elif "Elapsed (wall clock) time" in line:
            elapsed_time = parse_elapsed_time(line.split()[-1])
            if elapsed_time is not None:
                target_dict[key_prefix + "wall-clock-time"] = elapsed_time
        elif "Maximum resident set size" in line:
            target_dict[key_prefix + "max-rss"] = int(line.split()[-1])

    return target_dict

def accuracy(num_true_positives, num_false_positives, num_true_negatives, num_false_negatives):

    denominator = num_true_positives + num_false_positives + num_true_negatives + num_false_negatives
    if denominator == 0:
        return 0
    return (num_true_positives + num_true_negatives) / denominator

def P4_metric(num_true_positives, num_false_positives, num_true_negatives, num_false_negatives):

    denominator = 4.0 * num_true_positives * num_true_negatives + \
        (num_true_positives + num_true_negatives) * (num_false_positives + num_false_negatives)
    if denominator == 0:
        return 0
    return (4.0 * num_true_positives * num_true_negatives) / denominator

def positive_predictive_value(num_true_positives, num_false_positives, num_true_negatives, num_false_negatives):

    denominator = num_true_positives + num_false_positives
    if denominator == 0:
        return 0
    return num_true_positives / denominator

def true_positive_rate(num_true_positives, num_false_positives, num_true_negatives, num_false_negatives):

    denominator = num_true_positives + num_false_negatives
    if denominator == 0:
        return 0
    return num_true_positives / denominator

def true_negative_rate(num_true_positives, num_false_positives, num_true_negatives, num_false_negatives):

    denominator = num_true_negatives + num_false_positives
    if denominator == 0:
        return 0
    return num_true_negatives / denominator

def negative_predictive_value(num_true_positives, num_false_positives, num_true_negatives, num_false_negatives):

    denominator = num_true_negatives + num_false_negatives
    if denominator == 0:
        return 0
    return num_true_negatives / denominator

def get_classification_scores(predicted_labels, labels, *, target_dict=None, key_prefix=""):

    predicted_labels, labels = np.asarray(predicted_labels), np.asarray(labels)

    num_true_positives  = np.sum((predicted_labels == 1.0) & (labels == 1.0))
    num_false_positives = np.sum((predicted_labels == 1.0) & (labels == 0.0))
    num_true_negatives  = np.sum((predicted_labels == 0.0) & (labels == 0.0))
    num_false_negatives = np.sum((predicted_labels == 0.0) & (labels == 1.0))

    num_total = num_true_positives + num_false_positives + num_true_negatives + num_false_negatives
    assert num_total == len(labels)

    if target_dict is None:
        target_dict = {}

    target_dict[key_prefix + "accuracy"] = \
        accuracy(num_true_positives, num_false_positives, num_true_negatives, num_false_negatives)
    target_dict[key_prefix + "P4-metric"] = \
        P4_metric(num_true_positives, num_false_positives, num_true_negatives, num_false_negatives)
    target_dict[key_prefix + "ppv"] = \
        positive_predictive_value(num_true_positives, num_false_positives, num_true_negatives, num_false_negatives)
    target_dict[key_prefix + "tpr"] = \
        true_positive_rate(num_true_positives, num_false_positives, num_true_negatives, num_false_negatives)
    target_dict[key_prefix + "tnr"] = \
        true_negative_rate(num_true_positives, num_false_positives, num_true_negatives, num_false_negatives)
    target_dict[key_prefix + "npv"] = \
        negative_predictive_value(num_true_positives, num_false_positives, num_true_negatives, num_false_negatives)

    return target_dict
