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


def load_lgb_dataset(data_filename, label_filename):

    data = load_dataset_bin(data_filename)
    label = load_dataset_bin(label_filename)
    return lgb.Dataset(np.asarray(data), np.asarray(label) == 1.0)

# task = predict
# data = binary.test
# input_model= LightGBM_model.txt

# task type, support train and predict
# task = train

# # boosting type, support gbdt for now, alias: boosting, boost
# boosting_type = gbdt

# # application type, support following application
# # regression , regression task
# # binary , binary classification task
# # lambdarank , LambdaRank task
# # alias: application, app
# objective = binary

# # eval metrics, support multi metric, delimited by ',' , support following metrics
# # l1
# # l2 , default metric for regression
# # ndcg , default metric for lambdarank
# # auc
# # binary_logloss , default metric for binary
# # binary_error
# metric = binary_logloss,auc

# # frequency for metric output
# metric_freq = 1

# # true if need output metric for training data, alias: tranining_metric, train_metric
# is_training_metric = true

# # column in data to use as label
# label_column = 0

# # number of bins for feature bucket, 255 is a recommend setting, it can save memories, and also has good accuracy.
# max_bin = 255

# # training data
# # if existing weight file, should name to "binary.train.weight"
# # alias: train_data, train
# data = binary.train

# # validation data, support multi validation data, separated by ','
# # if existing weight file, should name to "binary.test.weight"
# # alias: valid, test, test_data,
# valid_data = binary.test

# # number of trees(iterations), alias: num_tree, num_iteration, num_iterations, num_round, num_rounds
# num_trees = 100

# # shrinkage rate , alias: shrinkage_rate
# learning_rate = 0.1

# # number of leaves for one tree, alias: num_leaf
# num_leaves = 63

# # type of tree learner, support following types:
# # serial , single machine version
# # feature , use feature parallel to train
# # data , use data parallel to train
# # voting , use voting based parallel to train
# # alias: tree
# tree_learner = serial

# # number of threads for multi-threading. One thread will use each CPU. The default is the CPU count.
# # num_threads = 8

# # feature sub-sample, will random select 80% feature to train on each iteration
# # alias: sub_feature
# feature_fraction = 0.8

# # Support bagging (data sub-sample), will perform bagging every 5 iterations
# bagging_freq = 5

# # Bagging fraction, will random select 80% data on bagging
# # alias: sub_row
# bagging_fraction = 0.8

# # minimal number data for one leaf, use this to deal with over-fit
# # alias : min_data_per_leaf, min_data
# min_data_in_leaf = 50

# # minimal sum Hessians for one leaf, use this to deal with over-fit
# min_sum_hessian_in_leaf = 5.0

# # save memory and faster speed for sparse feature, alias: is_sparse
# is_enable_sparse = true

# # when data is bigger than memory size, set this to true. otherwise set false will have faster speed
# # alias: two_round_loading, two_round
# use_two_round_loading = false

# # true if need to save data to binary file and application will auto load data from binary file next time
# # alias: is_save_binary, save_binary
# is_save_binary_file = false

# # output model file
# output_model = LightGBM_model.txt

def main(train_data_filename, train_label_filename, test_data_filename, test_label_filename,
         model_filename, num_estimators, max_tree_depth, num_threads):

    train_set = load_lgb_dataset(train_data_filename, train_label_filename)

    params = {
        "boosting_type": "gbdt",
        "objective": "binary",
        "metric": "binary_logloss",
        "num_trees": num_estimators,
        "learning_rate": 0.1,
        "num_leaves": 16384,
        "tree_learner": "serial",
        "num_threads": num_threads,
        "max_depth": max_tree_depth,
        "min_data_in_leaf": 1
    }

    model = lgb.train(params, train_set)
    model.save_model(model_filename)

    # model = lgb.LGBMClassifier(learning_rate=0.09,colsample_bytree=np.sqrt(len(data_points))/len(data_points), num_leaves=1024)
    # model = lgb.LGBMClassifier(boosting_type="dart",learning_rate=0.4,num_leaves=4096)
    # model.fit(train_data_points, train_labels, eval_metric='logloss')

    # print(dir(random_forest))
    # print(dir(random_forest.estimators_))
    # print(len(random_forest.estimators_))
    # print(random_forest.estimators_[0].get_depth())
    # print(dir(random_forest.estimators_[0]))
    # print(dir(random_forest.estimators_[0].tree_.node_count))
    # # print("Nodes", random_forest.tree_.node_count)

    # print("max-tree-depth", max([estimator.get_depth() for estimator in random_forest.estimators_]))
    # print("max-node-count", max([estimator.tree_.node_count for estimator in random_forest.estimators_]))

    # print("train-accuracy {:.4f}".format(model.score(train_data_points, train_labels)))
    # print("test-accuracy {:.4f}".format(model.score(test_data_points, test_labels)))

    test_data = load_dataset_bin(test_data_filename)
    labels = load_dataset_bin(test_label_filename)
    predicted_labels = np.round(model.predict(np.asarray(test_data)))
    accuracy = 1.0 - np.sum(np.asarray(predicted_labels) != np.asarray(labels)) / len(labels)
    print(f"test-accuracy {accuracy:.4f}")


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
