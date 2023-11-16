#!/usr/bin/env python3

import pickle
import sys

from sklearn.metrics import accuracy_score


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


def main(filename):

    points, labels = load_dataset_bin(filename)

    with open("sklearn.forest", "rb") as inf:
        random_forest = pickle.load(inf)

    yp = random_forest.predict(points)

    score = accuracy_score(labels, yp)
    print(f"test-accuracy {score:.4f}")


if __name__ == "__main__":

    main(*sys.argv[1:])
