#!/usr/bin/env python3

import pickle
import sys

from sklearn.metrics import accuracy_score


def load_dataset_bin(filename):

    import struct

    with open(filename, "rb") as inf:
        num_columns, = struct.unpack("<I", inf.read(4))
        data_points, labels = [], []
        unpacker = struct.Struct("<" + "f" * num_columns)
        for row in unpacker.iter_unpack(inf.read()):
            data_points.append(row[:-1])
            labels.append(row[-1])
    return data_points, labels


def main(filename):

    data_points, labels = load_dataset_bin(filename)

    with open("sklearn.forest", "rb") as inf:
        random_forest = pickle.load(inf)

    predicted_labels = random_forest.predict(data_points)

    score = accuracy_score(labels, predicted_labels)
    print(f"test-accuracy {score:.4f}")


if __name__ == "__main__":

    main(*sys.argv[1:])
