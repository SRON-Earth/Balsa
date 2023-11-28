#!/usr/bin/env python3

import argparse
import lightgbm as lgb
import numpy as np


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


def store_dataset_bin(filename, data):

    assert data.ndim >= 1 and data.ndim <= 2 and data.dtype == np.float32
    num_columns = 1 if data.ndim == 1 else data.shape[-1]
    with open(filename, "wb") as outf:
        outf.write(int.to_bytes(num_columns, length=4, byteorder="little"))
        for i in range(len(data)):
            outf.write(data[i].tobytes())


def main(model_filename, data_filename, label_filename):

    model = lgb.Booster(model_file=model_filename)
    data_points = load_dataset_bin(data_filename)
    predicted_labels = np.round(model.predict(np.asarray(data_points)))
    store_dataset_bin(label_filename, predicted_labels.astype(np.float32))


def parse_command_line_arguments():

    parser = argparse.ArgumentParser(description="Classify data using a pre-trained LightGBM classifier.")
    parser.add_argument("model_filename", metavar="MODEL_INPUT_FILE")
    parser.add_argument("data_filename", metavar="DATA_INPUT_FILE")
    parser.add_argument("label_filename", metavar="LABEL_OUTPUT_FILE")
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_command_line_arguments()
    main(**dict(vars(args)))
