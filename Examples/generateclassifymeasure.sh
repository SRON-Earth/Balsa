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
                feature = gaussian(122, 11  );
                feature = gaussian(40 , 9   );
                feature = gaussian(13 , 12  );
                feature = uniform( 100, 150 );
    }
    source(40)
    {
                feature = gaussian(12, 11  );
                feature = gaussian(40 , 9   );
                feature = gaussian(122 , 12  );
                feature = uniform( 100, 150 );
    }
   source(20)
    {
                feature = gaussian(122, 11  );
                feature = gaussian(40 , 9   );
                feature = gaussian(3 , 12  );
                feature = uniform( 1400, 150 );
    }
    source(20)
    {
                feature = gaussian(120, 10  );
                feature = gaussian(120, 10  );
                feature = gaussian( 30, 10  );
                feature = uniform( 140, 155 );
    }
}
EOF

# Generate a training set.
echo "Generating training data..."
balsa_generate -s 0 testgenscript.txt testgen-training-data.balsa testgen-training-labels.balsa

# Train the model on the training set.
echo "Training a model on the test data..."
balsa_train testgen-training-data.balsa testgen-training-labels.balsa testgen-model.balsa

# Generate a test test.
echo "Generating out-of-bag test data..."
balsa_generate -s 1 testgenscript.txt testgen-test-data.balsa testgen-test-labels.balsa

# Classify the in-bag test data set.
echo "Classifying the in-bag data..."
balsa_classify testgen-model.balsa testgen-training-data.balsa testgen-inbag-predictions.balsa

# Classify the out-of-bag test data.
echo "Classifying the out-of-bag test data..."
balsa_classify testgen-model.balsa testgen-test-data.balsa testgen-outofbag-predictions.balsa

# Evaluate the performance on the in-bag data.
echo "Evaluating classifier performance on in-bag data..."
balsa_metrics testgen-training-labels.balsa testgen-inbag-predictions.balsa

# Evaluate the performance on the in-bag data.
echo "Evaluating classifier performance on out-of-bag data..."
balsa_metrics testgen-test-labels.balsa testgen-outofbag-predictions.balsa




