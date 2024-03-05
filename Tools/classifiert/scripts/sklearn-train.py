import argparse
import numpy as np
import pathlib
import pickle
import struct
import sys
import time

from sklearn.ensemble import RandomForestClassifier

def load_dataset_bin(filename):

    header_format = "<I"
    header_size = struct.calcsize(header_format)
    with open(filename, "rb") as infile:
        num_columns, = struct.unpack(header_format, infile.read(header_size))
        dataset = np.frombuffer(infile.read(), "<f4")
        dataset.shape = (-1, num_columns)
    return dataset

def main(data_filename, label_filename, model_filename, num_estimators, max_tree_depth, num_threads):

    start_time = time.time()
    data_points = load_dataset_bin(data_filename)
    labels = load_dataset_bin(label_filename)
    end_time = time.time()
    data_load_time = end_time - start_time

    start_time = time.time()
    random_forest = RandomForestClassifier(n_estimators=num_estimators,
                                           n_jobs=num_threads,
                                           max_depth=max_tree_depth,
                                           max_features="sqrt",
                                           min_samples_leaf=1,
                                           min_samples_split=2,
                                           bootstrap=False)
    assert labels.shape == (len(data_points), 1)
    labels.shape = (-1,)
    random_forest.fit(data_points, labels)
    end_time = time.time()
    training_time = end_time - start_time

    start_time = time.time()
    with open(model_filename, "wb") as outf:
        pickle.dump(random_forest, outf)
    end_time = time.time()
    model_store_time = end_time - start_time

    print("Data Load Time: ", data_load_time)
    print("Training Time: ", training_time)
    print("Model Store Time:", model_store_time)
    print("Maximum Depth:", max([estimator.get_depth() for estimator in random_forest.estimators_]))
    print("Maximum Node Count:", max([estimator.tree_.node_count for estimator in random_forest.estimators_]))

def parse_command_line_arguments():

    def positive_integer(text):
        value = int(text)
        if value <= 0:
            raise ValueError
        return value

    parser = argparse.ArgumentParser(description="Train an sklearn classifier.")
    parser.add_argument("data_filename", type=pathlib.Path, metavar="DATA_INPUT_FILE")
    parser.add_argument("label_filename", type=pathlib.Path, metavar="LABEL_INPUT_FILE")
    parser.add_argument("model_filename", type=pathlib.Path, metavar="MODEL_OUTPUT_FILE")
    parser.add_argument("-d", "--max-tree-depth", type=positive_integer)
    parser.add_argument("-e", "--num-estimators", type=positive_integer, default="150")
    parser.add_argument("-t", "--num-threads", type=positive_integer, default="1")
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_command_line_arguments()
    main(**dict(vars(args)))
