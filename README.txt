--------------------------------------------------------------------------------
 ABOUT
--------------------------------------------------------------------------------

This repository contains the (as of yet unnamed) binary classification software
of SRON. This document contains a minimal introduction to the jargon, and an
overview and usage instructions of the software in this package.

--------------------------------------------------------------------------------
 INTRODUCTION
--------------------------------------------------------------------------------

Binary classification is the process of applying one of two labels to a
set of data points. Each data point consists of a fixed number of numeric
values, which are referred to as 'features'.

As an example, consider the output of a machine that scans pieces of fruit on a
factory conveyor belt. The machine weighs each item, and registers the average
RGB colour value of the pixels in a digital photo of the piece of fruit. In this
example, each data point has four features: the weight, and the red-, green- and
blue components.

    +-----------------------------+
    | Weight | Red | Green | Blue |
    +-----------------------------+
    | 133    | 255 | 204   | 102  |
    | 88     | 152 | 255   | 51   |
    | 163    | 250 | 152   | 99   |
    | 99     | 248 | 199   | 89   |
    +-----------------------------+

The binary classification problem in this example is to distinguish between
apples and oranges. These are examples of 'labels' in machine learning.

Labeling can be achieved by training a machine learning tool on a set of data
points for which the labels are already known. A 'training set' is a data set as
the one shown above, plus a corresponding set of labels ('apple' or 'orange')
for each point. In a binary classification problem there are only two different
labels. By convention, the actual labels are integer values rather than text.

A label set for the example data might look as follows:

    +--------+
    | Apple? |
    +--------+
    | 0      |
    | 1      |
    | 0      |
    | 1      |
    +--------+

A training set can be used to train a predictive model. If the model is good
enough, it can then be used to label data points for which the labels are not
known ('prediction'). To test the quality of a model, it is common to split the
available training set into a proper training set and a 'test set'. The training
set is used to train the model, the test set is then used to determine the
quality of the model. In this scenario, the model is used to make 'out of bag
predictions' (i.e. the model is used on data points that were not part of the
training set). Various measures can then be used to assess the quality of the
model. The simplest one would be the 'accuracy', which is the number of accurate
predictions of the test data dividded bythe total size of the test data.

In addition to the predictive quality of a trainer/classifier combination, the
merits of a machine learning solution can be judged in terms of the amount of
computational resources it uses. The 'peak memory usage' is the amount of
physical or virtual memory an application uses during its runtime. The 'wall
clock time' is the total time used, and the 'CPU time' is the sum of the time
used by all CPU cores that are in use by the application. For applications where
data sets are large and/or response times are short, these aspects become
significant factors that affect practical usability of both trainers and
classifiers.

--------------------------------------------------------------------------------
 CONTENTS
--------------------------------------------------------------------------------

This repository contains the following software:

 - rf_train: An application that can train binary Random Forest Classifiers on
   an appropriate training data set. The output is a model file.

 = rf_classify: An application that uses the model files of rf_train to do
   predictions on unlabeled data points.

 - classifiert: A tool that can be used to compare the predictive quality and
   the system resource usage of various trainers and classifiers, and to split
   large training sets into proper training- and test sets.


--------------------------------------------------------------------------------
 REQUIREMENTS
--------------------------------------------------------------------------------

The rf_train/rf_classify tools require a C++ compiler and a CMake installation.
The exact version requirements of CMake and the C++ standard are subject to
change. The versions are checked by CMake during the build process.

The classifiert tool requires Python 3.

There are no specific operating system requirements to the software.

--------------------------------------------------------------------------------
 FILE FORMATS
--------------------------------------------------------------------------------

Data sets for rf_train and rf_classify are ingested from a simple binary file
format, consisting of a 32-bit unsigned integer that indicates the feature count
'N' of each data point, followed by an arbitrary number of points. Each point
consists of N double precision floating point values, representing the features.
Floating point values are all encoded according to the IEEE 754 standard. All
numeric values are stored in little-endian order.

Training data sets consist of a data point file as described above, and a
separate label file. The label files have the same structure as the data files,
except that the number of 'features' is always one. Nonzero values are
interpreted as 'true' or 'label 1', zero values as 'false' or 'label 0'.

--------------------------------------------------------------------------------
 BUILDING
--------------------------------------------------------------------------------

From a shell that has CMake in its path, navigate to the root directory of this
package.

On UNIX systems (Linux, MacOS, etc.) execute the following commands:

    mkdir build
    cd build
    cmake ..
    make

For Windows systems execute the following commands:

    mkdir build
    cd build
    cmake -G "NMake Makefiles"
    nmake

If all steps are successful, two binaries are created:

   build/Sources/rf_train
   build/Sources/rf_classify

In case of failure, either CMake or the C++ compiler should give relevant
information about your installation that may have to be resolved before you can
proceed.

--------------------------------------------------------------------------------
 USAGE
--------------------------------------------------------------------------------

In order to use the rf_train/rf_classifier tools, one needs a set of appropriate
data files, as described in the FILE FORMAT section. Examples of such files are
provided in the Data/ directory.

Usage information of rf_train or rf_classify is available by running either tool
without any arguments.

The 'classifiert' tool can be run by executing the python interpreter in the
Tools directory, where the 'classifiert' Python package is located:

    cd Tools
    python -m classifiert

Further usage information is available by passing the '-h' option:

    python -m classifiert -h

--------------------------------------------------------------------------------
 SUPPORT
--------------------------------------------------------------------------------

Support queries may be directed to info@jigsaw.nl

--------------------------------------------------------------------------------
 END
--------------------------------------------------------------------------------


