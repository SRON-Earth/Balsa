#!/usr/bin/env python3

import argparse
import pickle
import sys

import lightgbm as lgb


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


def main(train_data_filename, train_label_filename, test_data_filename, test_label_filename,
         model_filename, num_estimators, max_tree_depth, num_threads):

    train_data_points = load_dataset_bin(train_data_filename)
    train_labels = load_dataset_bin(train_label_filename)

    test_data_points = load_dataset_bin(test_data_filename)
    test_labels = load_dataset_bin(test_label_filename)

    # model = lgb.LGBMClassifier(learning_rate=0.09,colsample_bytree=np.sqrt(len(data_points))/len(data_points), num_leaves=1024)
    model = lgb.LGBMClassifier(boosting_type="dart",learning_rate=0.4,num_leaves=4096)
    model.fit(train_data_points, train_labels, eval_metric='logloss')

    # random_forest = lgb.LGBMClassifier(boosting_type="rf", max_depth=max_tree_depth, n_estimators=num_estimators, n_jobs=num_threads)

    # random_forest = lgb.LGBMClassifier(boosting_type="rf",
    #                          num_leaves=165,
    #                          colsample_bytree=.5,
    #                          n_estimators=150,
    #                          min_child_weight=5,
    #                          min_child_samples=10,
    #                          subsample=.632, # Standard RF bagging fraction
    #                          subsample_freq=1,
    #                          min_split_gain=0,
    #                          reg_alpha=10, # Hard L1 regularization
    #                          reg_lambda=0,
    #                          n_jobs=3)

    # print(dir(random_forest))
    # print(dir(random_forest.estimators_))
    # print(len(random_forest.estimators_))
    # print(random_forest.estimators_[0].get_depth())
    # print(dir(random_forest.estimators_[0]))
    # print(dir(random_forest.estimators_[0].tree_.node_count))
    # # print("Nodes", random_forest.tree_.node_count)

    # print("max-tree-depth", max([estimator.get_depth() for estimator in random_forest.estimators_]))
    # print("max-node-count", max([estimator.tree_.node_count for estimator in random_forest.estimators_]))

    print("train-accuracy {:.4f}".format(model.score(train_data_points, train_labels)))
    print("test-accuracy {:.4f}".format(model.score(test_data_points, test_labels)))


def parse_command_line_arguments():

    def positive_integer(text):
        value = int(text)
        if value <= 0:
            raise ValueError
        return value

    parser = argparse.ArgumentParser(description="Train a LightGBM classifier.")
    parser.add_argument("train_data_filename", metavar="TRAIN_DATA_FILE")
    parser.add_argument("train_label_filename", metavar="TRAIN_LABEL_FILE")
    parser.add_argument("test_data_filename", metavar="TEST_DATA_FILE")
    parser.add_argument("test_label_filename", metavar="TEST_LABEL_FILE")
    parser.add_argument("model_filename", metavar="MODEL_OUTPUT_FILE")
    parser.add_argument("-d", "--max-tree-depth", type=positive_integer)
    parser.add_argument("-e", "--num-estimators", type=positive_integer, default="150")
    parser.add_argument("-t", "--num-threads", type=positive_integer, default="1")
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_command_line_arguments()
    main(**dict(vars(args)))
