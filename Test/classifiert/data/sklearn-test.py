#!/usr/bin/env python3

import pickle
import sys

from sklearn.metrics import accuracy_score


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


def main(data_filename, label_filename):

    data_points = load_dataset_bin(data_filename)
    labels = load_dataset_bin(label_filename)

    with open("sklearn.forest", "rb") as inf:
        random_forest = pickle.load(inf)

    predicted_labels = random_forest.predict(data_points)

    score = accuracy_score(labels, predicted_labels)
    print(f"test-accuracy {score:.4f}")


if __name__ == "__main__":

    main(*sys.argv[1:])
