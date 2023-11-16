#!/usr/bin/env python3

import pickle
import sys

from sklearn.ensemble import RandomForestClassifier


def load_dataset_bin(filename):

    import numpy as np
    import struct

    with open(filename, "rb") as inf:
        num_points, num_features = struct.unpack("<QI", inf.read(8 + 4))
        points = np.zeros((num_points, num_features), dtype=np.float32)
        labels = np.zeros((num_points,), dtype=np.uint8)
        unpacker = struct.Struct("<" + "f" * num_features + "B")
        for i, row in enumerate(unpacker.iter_unpack(inf.read())):
            points[i] = row[:-1]
            labels[i] = row[-1]
    return points, labels


def main(filename, nthreads=1):

    nthreads = int(nthreads)

    points, labels = load_dataset_bin(filename)
    random_forest = RandomForestClassifier(n_estimators=150,
                                           n_jobs=nthreads,
                                           max_depth=50,
                                           max_features="sqrt",
                                           min_samples_leaf=1,
                                           min_samples_split=2)
    random_forest.fit(points, labels)

    print("max-tree-depth", max([estimator.get_depth() for estimator in random_forest.estimators_]))
    print("max-node-count", max([estimator.tree_.node_count for estimator in random_forest.estimators_]))

    with open("sklearn.forest", "wb") as outf:
        pickle.dump(random_forest, outf)


if __name__ == "__main__":

    main(*sys.argv[1:])
