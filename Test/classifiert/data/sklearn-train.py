#!/usr/bin/env python3

import pickle
import sys

from sklearn.ensemble import RandomForestClassifier


def load_dataset_bin(filename):

    import struct

    with open(filename, "rb") as inf:
        num_features, = struct.unpack("<I", inf.read(4))
        data_points, labels = [], []
        unpacker = struct.Struct("<" + "f" * (num_features + 1))
        for row in unpacker.iter_unpack(inf.read()):
            data_points.append(row[:-1])
            labels.append(row[-1])
    return data_points, labels


def main(filename, nthreads=1):

    nthreads = int(nthreads)

    data_points, labels = load_dataset_bin(filename)
    random_forest = RandomForestClassifier(n_estimators=150,
                                           n_jobs=nthreads,
                                           max_depth=50,
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
