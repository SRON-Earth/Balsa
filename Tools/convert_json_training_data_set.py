#!/usr/bin/env python3

import jsonpickle
import jsonpickle.ext.numpy
import numpy as np
import os
import struct
import sys

def main(filename_in, filename_out):

    jsonpickle.ext.numpy.register_handlers()
    with open(filename_in) as json_file:
        points, labels = jsonpickle.decode(json_file.read())

    num_points, num_features = points.shape
    print("No. of points: ", num_points)
    print("No. of features: ", num_features)
    percentage_zeros = 100.0 * np.sum(labels == 0) / labels.size
    percentage_ones = 100.0 * np.sum(labels == 1) / labels.size
    print(f"Labels: 0 ({percentage_zeros:.2f}%), 1 ({percentage_ones:.2f}%)")
    assert np.sum(labels == 0) + np.sum(labels == 1) == labels.size

    labels = labels.astype(np.uint8)

    assert points.dtype == np.float32
    assert labels.dtype == np.uint8
    tuple_value_types = "f" * num_features + "B"

    with open(filename_out, "wb") as outf:

        outf.write(struct.pack("<I", num_features + 1))
        outf.write(tuple_value_types.encode("ascii"))

        for i in range(num_points):
            outf.write(points[i].tobytes())
            outf.write(labels[i].tobytes())

if __name__ == "__main__":
    
    if len(sys.argv) != 3:
        program_name = os.path.basename(sys.argv[0])
        print(f"Usage: {program_name} INPUT OUTPUT")
        sys.exit(1)

    main(*sys.argv[1:])
