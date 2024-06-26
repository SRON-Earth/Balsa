#!/usr/bin/env bash

# This script generates test data, trains a random forest on it, classifies it,
# and prints the performance metrics after using the trained model on the in-bag
# test data. This script looks for Balsa command-line tools in the current
# directory, so it can be run from the build/Sources/ directory.

# Add the current directory to the path, in case this script is run from the build directory without installing.
PATH=$PATH:.

# Create a test set generation script.
echo "Generating a test data generator script..."
cat << EOF > testgenscript.txt
multisource(4)
{
    source(40)
    {
        gaussian(245, 40  );
        gaussian(188 ,40  );
        gaussian(66 , 40  );
        uniform( 90 , 115 );
    }
    source(40)
    {
        gaussian(158, 30 );
        gaussian(245, 30  );
        gaussian(66 , 30  );
        uniform( 95 , 150 );
    }
    source(20)
    {
        gaussian(245, 50  );
        gaussian(123, 40  );
        gaussian(66 , 40  );
        uniform( 90 , 140 );
    }
    source(20)
    {
        gaussian( 71,  40  );
        gaussian( 39,  40  );
        gaussian( 112, 40  );
        uniform(  90 , 155 );
    }
}
EOF

POINTCOUNT=10000
THREADCOUNT=10
TREECOUNT=400
IMPORTANCEPOINTCOUNT=1000
IMPORTANCEREPEATS=10

# Generate a training set.
echo "Generating training data..."
balsa_generate -p ${POINTCOUNT} -s 0 testgenscript.txt testgen-training-data.balsa testgen-training-labels.balsa

# Train the model on the training set.
echo "Training a model on the test data..."
balsa_train -t ${THREADCOUNT} -c ${TREECOUNT} testgen-training-data.balsa testgen-training-labels.balsa testgen-model.balsa

# Generate a test test.
echo "Generating out-of-bag test data..."
balsa_generate -p ${POINTCOUNT} -s 1 testgenscript.txt testgen-test-data.balsa testgen-test-labels.balsa

# Classify the in-bag test data set.
echo "Classifying the in-bag data..."
balsa_classify testgen-model.balsa testgen-training-data.balsa
mv testgen-training-data-predictions.balsa testgen-inbag-predictions.balsa

# Classify the out-of-bag test data.
echo "Classifying the out-of-bag test data..."
balsa_classify testgen-model.balsa testgen-test-data.balsa
mv testgen-test-data.balsa testgen-outofbag-predictions.balsa

# Evaluate the performance on the in-bag data.
echo "Evaluating classifier performance on in-bag data..."
balsa_measure testgen-training-labels.balsa testgen-inbag-predictions.balsa

# Evaluate the performance on the in-bag data.
echo "Evaluating classifier performance on out-of-bag data..."
balsa_measure testgen-test-labels.balsa testgen-outofbag-predictions.balsa

# Generate a test test.
echo "Generating test data for feature importance evaluation..."
balsa_generate -p ${IMPORTANCEPOINTCOUNT} -s 1 testgenscript.txt testgen-fimp-data.balsa testgen-fimp-labels.balsa

# Evaluate feature importance.
echo "Evaluating feature-importance of the classifier..."
balsa_featureimportance -r ${IMPORTANCEREPEATS} testgen-model.balsa testgen-fimp-data.balsa testgen-fimp-labels.balsa

