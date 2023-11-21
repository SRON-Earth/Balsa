#!/usr/bin/env python3

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


def main(data_filename, label_filename, num_estimators, max_tree_depth, num_threads):

    num_estimators = int(num_estimators)
    max_tree_depth = None if max_tree_depth == "None" else int(max_tree_depth)
    num_threads = int(num_threads)

    data_points = load_dataset_bin(data_filename)
    labels = load_dataset_bin(label_filename)

    random_forest = RandomForestClassifier(n_estimators=num_estimators,
                                           n_jobs=num_threads,
                                           max_depth=max_tree_depth,
                                           max_features="sqrt",
                                           min_samples_leaf=1,
                                           min_samples_split=2)
    random_forest.fit(data_points, labels)

    print("max-tree-depth", max([estimator.get_depth() for estimator in random_forest.estimators_]))
    print("max-node-count", max([estimator.tree_.node_count for estimator in random_forest.estimators_]))

    with open("sklearn.forest", "wb") as outf:
        pickle.dump(random_forest, outf)


if __name__ == "__main__":

    main(*sys.argv[1:])
