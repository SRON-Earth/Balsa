#!/usr/bin/env python3

import argparse
import pickle
import sys

from sklearn.ensemble import RandomForestClassifier


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


def main(data_filename, label_filename, model_filename, num_estimators, max_tree_depth, num_threads):

    data_points = load_dataset_bin(data_filename)
    labels = load_dataset_bin(label_filename)

    random_forest = RandomForestClassifier(n_estimators=num_estimators,
                                           n_jobs=num_threads,
                                           max_depth=max_tree_depth,
                                           max_features="sqrt",
                                           min_samples_leaf=1,
                                           min_samples_split=2,
                                           bootstrap=False)
    random_forest.fit(data_points, labels)

    print("max-tree-depth", max([estimator.get_depth() for estimator in random_forest.estimators_]))
    print("max-node-count", max([estimator.tree_.node_count for estimator in random_forest.estimators_]))

    with open(model_filename, "wb") as outf:
        pickle.dump(random_forest, outf)


def parse_command_line_arguments():

    def positive_integer(text):
        value = int(text)
        if value <= 0:
            raise ValueError
        return value

    parser = argparse.ArgumentParser(description="Train an sklearn classifier.")
    parser.add_argument("data_filename", metavar="DATA_INPUT_FILE")
    parser.add_argument("label_filename", metavar="LABEL_INPUT_FILE")
    parser.add_argument("model_filename", metavar="MODEL_OUTPUT_FILE")
    parser.add_argument("-d", "--max-tree-depth", type=positive_integer)
    parser.add_argument("-e", "--num-estimators", type=positive_integer, default="150")
    parser.add_argument("-t", "--num-threads", type=positive_integer, default="1")
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_command_line_arguments()
    main(**dict(vars(args)))
