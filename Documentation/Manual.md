![](Images/balsalogo.png)

# Balsa: A Fast C++ Random Forest Classifier

## About

Balsa is a memory- and CPU-efficient C++ implementation of the RandomForest classification algorithm. The Balsa package contains the following items:

*  The command-line tools "balsa\_train" and "balsa\_classify" for training and classification.
*  The "balsa" C++ library for training and classification directly from within a third-party C++ application. 
*  The "classifiert" tool, a utility for manipulating test data sets, and for evaluating the performance of Balsa and other classifiers. 

## Current Limitations

This version of Balsa is limited to binary classification problems. Multi-label classification is planned for a future update.

## Theory

This section provides a brief overview of the theory of classification problems. It introduces the basic jargon (in *emphasis*) that is used and assumed to be known throughout the rest of the documentation.

### Classification Problems

*Classification* is the process of assigning one of a fixed number of *labels* to a set of data points. Each data point consists of a fixed number of numeric values, known as *features*.

As an example, consider the output of a machine that scans pieces of fruit on a factory conveyor belt. The machine weighs each item, and registers the average RGB colour value of the pixels in a digital photo of the piece of fruit. Each data point has four features: the weight, and the average red, green, and blue components of the image pixels.

| Weight | Red | Green | Blue |
|--------|-----|-------|------|
| 133    | 255 | 204   | 102  |
| 88     | 152 | 255   | 51   |
| 163    | 250 | 152   | 99   |
| 99     | 248 | 199   | 89   |

Each piece of fruit has to be labeled as either and apple or an orange, based on the measurement data. Since there are only two possible labels ('apple' or 'orange') this is an example of a *binary classification problem*. 

### Training

The classification problem can be solved by *training* a machine learning tool on a set of data points for which the labels are already known. The training process creates a *predictive model* (or simply *model*) that can predict the labels for points that were not part of the training data.

A *training set* is a data set as the one shown earlier, plus a corresponding set of known labels ('apple' or 'orange') for each point. 
The training set has to be obtained from a source that is known to be correct, or as close to correct as possible. In this case, it could have been a human observer that wrote down the correct labels for 1000 pieces of fruit. If, as in this example, the training set is likely to be completely correct, it is often called the *ground truth*.

Labels are usually represented as integers in machine learning, because their semantic interpretation is not important to most machine learning algorithms. Using 0 to identify oranges and 1 to identify apples, the ground truth for the example data points might be as follows:

| Apple? |
|--------|
| 0      |
| 1      |
| 0      |
| 1      |

### Model Quality

To test the predictive quality of a model, the available training set can be split into a proper training set and a *test set*. The training set is used to train the model, the test set is then used to determine the quality of the model. The predictions that the model makes on the points in the test set are called *out-of-bag* predictions (OOB).

There are various metrics for the quality of the model. Arguably the simplest and most used metric is the *accuracy*, which is the number of correct predictions made by the model, divided by the total number of points for which a prediction was made.

There are several other common metrics for binary classifiers:

* The *Positive Predictive Value* or *PPV*, also *precision*, is the probability that the ground truth is positive ("true") when the classifier returns a positive label.
* The *True Positive Rate* or *TPR*, also *sensitivity*, is the probability that the classifier returns a positive label when the ground truth is positive.
* The *True Negative Rate* or *TNR*, also *specificity*, is the probability that the classifier returns a negative label ("false") when the ground truth is negative.
* The *Negative Predictive Value* or *NPV* is the probability that the ground truth is negative when the classifier returns a negative label.

The *P4 metric* is calculated by taking the harmonic mean of the PPV, TPR, TNR, and NPV:

	P4 = 4 / ( (1/p) + (1/r) + (1/s) + (1/n) )

### Resource Usage

Any computer software uses scarce system resources during its use: storage space, system memory (RAM), and processor time (CPU time). Either one of these resources can become a practical bottleneck during training or classification when large datasets are used. As a result, a machine-learning solution can become unusably slow on a particular system.

The following metrics are useful to evaluate the resource usage of a particular run of a particular program:

* Wall-clock Time: the total amount of time it takes to complete the run.
* Total CPU Time: the total amount of time the program was active on a CPU, summed over all available CPUs. 
* Peak Memory Usage: the largest amount of RAM the program used during its run.

N.B. the relation between walll-clock time and total CPU-time is of particular interest, because it gives some information about how effectively the program uses its available CPUs. If a program runs for 10 minutes (wall-clock time) on 8 cores, it can measure at most 80 minutes of total CPU time. If the total is less than that, not all cores could be kept busy for the full 10 minutes.

Some care has to be taken when interpreting these metrics, however. As an example, consider a system has 10 separate cores, and various machine-learning programs that classify a large data set. Here are the metrics for these programs (times in minutes, memory in gigabytes):

| Program | Wall-Clock Time | CPU-Time | Peak Memory |
|---------|-----------------|----------|-------------|
| Alpha   | 10              | 100      | 32          |
| Beta    | 20              | 18       | 10          |
| Gamma   | 11              | 22       | 32          |
| Delta   | 10              | 10       | 128         |

Alpha and Delta are the clear winners in terms of wall-clock turnaround time. Alpha needs all 10 available cores to achieve this. Delta only uses one core, but it uses an excessive amount of memory. While Gamma is only slightly slower than the top contenders, it only uses two cores and a moderate amount of memory. 

Beta is slowest, but it has a very light memory footprint. It is likely that it is a single-threaded (single-core) solution, because it uses less CPU time than wall-clock time. This means that it might be possible to speed up Beta by making it multi-threaded. The fact that its CPU time is *less* than the wall-clock time means that Beta is likely spending some of its time waiting for disk I/O operations. This could be the reason why it its memory usage is so low. Which of these solutions is best depends on the particular demands of the application.

## Using Balsa

### Prerequisites

The core Balsa command-line tools and C++ library have the following system prerequisites:

* A C++ compiler.
* The CMake build system.

N.B. version requirements for both the C++ standard and the CMake build system are checked by the CMake build files. 

The following optional prerequisites apply:

* To use the optional *classifiert* utility, Python 3 is required.
* For command-line builds on Windows (recommended), Microsoft's 'NMake' utility is recommended.
* A Markdown viewer or Markdown-to-HTML conversion tool is recommended for reading this manual.

### Installation

On UNIX systems (Mac, Linux) run the following commands from the top-level source directory to build Balsa:

```
mkdir build
cd build
cmake ..
make
```

The equivalent procedure for Windows (using NMake) is to open a Visual Studio Command Prompt in the top-level source directory, and to execute the following commands:

```
mkdir build
cd build
cmake -G "NMake Makefiles" ..
nmake
```

After building the software, it can be installed using `make install` (UNIX) or `nmake install` (Windows).

### Using The Command-line Tools

The Balsa package provides stand-alone command-line tools for training and classification. We recommend to explore these tools before integrating Balsa directly into a C++ application.

#### Input File Format

The command-line tools read training data and/or classification data from files that adhere to a simple binary format. All input files consist of one 32-bit integer, indicating the *row length* of the rest of the file. Following the row length are an arbitrary number of rows. Each row is a block of double-precision floating point values (8-byte IEEE 754) of the specified row length. All values are in little-endian byte order.

For point data files, the row length indicates the number of features in each point. For label files, the row length is always equal to 1.

#### Training

To train a binary forest model on the command-line, run the following command, substituting the appropriate filenames (see [input file format)](#### Input File Format):

```
balsa_train yourinputfile yourlabelfile yourmodelfile
```

This command creates "yourmodelfile", a random forest model trained on "yourinputfile" and "yourlabelfile".

Various parameters of the training process can be controlled to make trade-offs between the disk usage of the generated model files, the processor/multi-core utilization, wall-clock time, etc. Running `balsa_train` without any arguments displays the full range of options:

```
Usage:

   balsa_train [options] <data input file> <label input file> <model output file>

 Options:

   -t <thread count>: Sets the number of threads (default is 1).
   -d <max depth>   : Sets the maximum tree depth (default is +inf).
   -c <tree count>  : Sets the number of trees (default is 150).
   -s <random seed> : Sets the random seed (default is a random value).
```
The thread count influences performance of the trainer. More threads will speed up the training process, at the expense of more cores and more peak memory usage. The maximum depth and tree count can be used to trade model size (and consequently classifier performance) for model quality.

#### Classification 

A model can be used to classify a data set as follows:

```
balsa_classify yourmodelfile yourdatafile youroutputfile
```
The output is written in the same format as the training label file.

The resource usage of the command-line classifier can be tuned to achieve to achieve shorter wall-clock times at the expense of additional memory usage and CPU load. We note, however, that the command-line classifier already peforms very well in single-core mode. 

The [performance optimization](## Performance Tuning) section explains the various trade-offs for performance optimization that apply both to the command-line classifier and the C++ API of Balsa. Running balsa_classify without parameters prints the options.

### Using The Balsa Library

It is possible to use the Balsa classifier directly from an existing C++ application. This section demonstrates various useful ways to interact with the library. 

Implementation note: The examples in this section assume that you have an example model file available. The model file must be trained on a data set of your choosing using the command-line tools. It is possible to train a model from within C++, but we do not document the training API at this point in the development process. 

#### Basic Classification

The following program demonstrates the simplest and most basic way to use the Balsa library. It uses an array of double-precision floating point numbers to store a set of points in memory, and an array of bools to store the output labels. 

```
#include <iostream>
#include <balsa.h>

int main( int, char ** )
{
	// Create a dataset with 4 points,  3 features per point.
	double points[] = { 0.0, 1.0, 2.0 , // Point 0
						0.0, 1.0, 2.0 , // Point 1
						0.0, 1.0, 2.0 , // Point 2
						0.0, 1.0, 2.0 } // Point 3
							
	// Allocate space for the labels.
	bool labels[4];
	
	// Create a classifier for 3 features, and classify the points.
	RandomForestClassifier classifier( "yourmodelfile.dat", 3 );
	classify( points, points + 12, labels );
	
	return 0;
}
```

This example can be compiled, linked and executed on a UNIX system as follows:

```
c++ -std=c++17 example1.cpp -L balsa -o example1
./example1
```

#### Using Different Containers

In the first example, the point data was stored in an array of doubles, and the output labels were written to an array of bools. While this is the most syntactically simple way to use Balsa, it has several drawbacks. First, a real-world application will often have its data stored in different types of containers, and copying it to a flat array before classification is inefficient. Second, it is error-prone to deal with explicit, hard-coded array sizes. 

Balsa provides some flexibility to choose input- and output containers. The RandomForestClassifier is templated on the iterators that point to the input data and the output data. For reasons of computational efficiency, there are some restrictions to the types of iterators that will work:

1. The input iterators must adhere to the `std::random_access_iterator` concept.
2. The input iterators must point to a floating-point type (double or float).
3. The input iterator must iterate over the points and features in row-major order.
4. The output iterator for labels must point to a bool-compatible type.

In practice, these restrictions mean that the point data in your application must be layed out consecutively in memory, in row-major order, as demonstrated in the first example, but the exact container that is used can vary.

The following program demonstrates how the classifier can be instantiated for different types of input- and output containers.

```
#include <balsa.h>
#include <vector>
#include <valarray>

int main( int, char ** )
{
	// Use a vector of doubles for input, and a valarray of ints for output.
	typedef std::vector<double> Points;
	typedef std::valarray<int>  Labels;
	Points points( 100 );
	Labels labels( 100 );
	// ... Fill point data ...
	
	// Create a classifier for these container types.
	RandomForestClassifier<Points::const_iterator,Labels::iterator >
							classifier( "yourmodelfile.dat", 10 ); // 10 features.

	// Classify the points.
	classifier.classify( points.begin(), points.end(), labels.begin() );
	
	return 0;
}
```

This example exposes the fact that RandomForestClassifier is a class-template, not a class. It needs to be instantiated for a particular type of container. The reason why this is the case (versus, for example, just making the classify() method a template-method) has to do with various performance optimizations and trade-offs within the core library. By default, as in the first example, the iterator types are `double *` and `bool *` for input and output.

### Using Single-Precision Features

By default, Balsa uses double-precision floating point numbers for feature values. For many applications this is overkill. It can therefore be beneficial to use single-precision floats, which would effectively halve the memory usage. Changing the precision is achieved implicitly by choosing a different input iterator type, in the same manner as in the previous example:

```
int main( int, char ** )
{
	// Use a vector of single-precision floats as input.
	typedef std::vector<float> Points;

	...
	
	// The iterator of the points container will propagate single-precision 
	// throughout the classifier.
	RandomForestClassifier<Points::const_iterator,Labels::iterator >
							classifier( "yourmodelfile.dat", 10 ); // 10 features.
							
	...
}
```

### Performance Tuning in C++

By default, the RandomForestClassifier does all of its work inside the main thread. The classify() method uses this thread to load the model from a file, and to perform the actual classification. It is possible to tell the classifier to use worker-threads to aid in classification, and it is possible to tune the overall memory usage of the classifier.

```
int main( int, char ** )
{
	...
	
	// A classifier for 2 features, using 3 worker threads, 
	// and max. 10 preloaded trees.
	RandomForestClassifier classifier( "model.dat", 2, 3, 10 );
	
	...
}
```

N.B. This example shows how to control the tuning parameters, but it does not explain what they do or how you should use them. We refer to the [Perfomance Tuning](## Performance Tuning) section for a general discussion, as it applies equally to both the command-line version and the C++ classifier.

## Performance Tuning

Balsa is designed to be a very fast, resource-efficient implementation of the Random Forest algorithm. In order to get the best possible performance out of it in terms of peak memory usage, wall-clock time and CPU utilization, it is necessary to have some insight into the inner workings of the library. This section provides a mental model for the classifier, and a number of practical tuning guidelines.

### Inner Classifier Workflow

The Random Forest algorithm is one example of the more general family of *ensemble classifiers*. In an enseble classifier, multiple simple/crude classifiers are used to classify the data points. The votes of each of these sub-classifiers are then added/weighted to arrive at a final vote for the label of each data point. In the Random Forest algorithm, the sub-models are randomized decision trees.

While Balsa only offers Random Forests at the moment, the underlying classifier architecture is quite generic. In order to understand and tune the performance, it is instructive to look at the inner workflow of the Ensemble Classifier implementation in Balsa. The following infographic shows its structure: 

![](Images/EnsembleClassificationWorkflow.png)

The image shows how the sub-models (usually decision trees) are loaded from disk into memory by the Model Loader. A Work Divider divides the loaded sub-models over the available Worker Threads. (In single-threaded mode, there are no Worker Threads, so the main thread does the work itself). The Worker Threads apply each sub-model to the data points, and they keep internal vote tables to accumulate the results. After all sub-models are processed, an Accumulator harvests and sums the vote tables of eacht thread, and produces the final label results.

### Optimizing Performance

With the inner workflow of the Ensemble Classifier in mind, there are several general observations to make with respect to performance optimization:

* **It is best to classify as many points as possible in one single "classify()"  call.** 

The classification algorithm is fundamentally faster in bulk-mode than it is on individual points, making this by far the most important optimization consideration of all. In any situation, it is best to classify as many points in a single call as you can.

* **Avoid loading the model more than once if you have the RAM.**

In some applications, new data points arrive as input for classification in a continuous stream from outside. If there is enough RAM available, it is best to load the model from disk only once for this type of scenario. This can be achieved by choosing the `model-preload` parameter in such a way that it allows all sub-models to be loaded into the Model Loader's internal memory.

If you have trained a forest of e.g. 150 trees, you can set the model preload parameter of RandomForestClassifier to 150. The first call to classify() will then load all the trees into RAM. A second call (for a batch of points that arrives later) will be much faster than the first. If your application is a server that stays resident to process new batches of points that periodically come in, this will significantly reduce the wall-clock time. Model-loading is typically much slower than the classification itself.

* **If the model is only used once per run, preloading too much will only waste RAM.**

At a minimum, the Ensemble Classifier will preload one model per worker thread, even if you set the preload parameter to 1. If your application does not allow you to process multiple incoming batches of points as per the previous bullet point, there is no benefit in preloading the entire model. It will only consume RAM. In this case, it is best to leave the setting to its default value.

Note that this case applies to the command-line balsa_classify as well. The command-line classifier can only use the model once to process all input points. It keeps its memory footprint very low by *not* preloading the model, lazily loading new trees to keep the workers busy.

* **If the model cannot be re-used, multithreading is of limited use.**

The classifier can be used in single- or multi-threaded mode. In applications where the loaded model cannot be re-used on multiple incoming batches of points, multithreading is of limited use, because model-loading will take up much more time than classification. Adding more worker-threads will help, but to a limited extent, because model-loading cannot benefit from it. 

It is straightforward to predict the effect of worker threads in a load-once-classify-once scenario:

1. If zero worker threads are used, the wall-clock time will consist of a long model-load period, followed by a shorter classification period.
2. If one worker thread is used, the wall-clock time will be only slightly longer than the model-load time, as classification is largely done by the worker-thread while the model loader is waiting for the disk.
3. If more worker threads are used, there may be minimal further improvement in the classification time. In most cases the threads will starve while waiting for the model loader. This is visible as a CPU time that is much lower than wall-clock time multiplied by thread count.

A more positive (and correct) perspective is that the Balsa classifier is so fast that it is hardly necessary to use multiple cores in most applications.

In conclusion:

* **The ideal performance scenario for a minimal peak-memory footprint** is to use the classifier with either _zero or one worker threads_, and a _model preload maximum of 1_. This (zero workers) is the default configuration of the classifier in both the command-line and the library version.
* **The ideal performance scenario for minimal classification wall-clock time** is _to preload the entire model in RAM_ (set preload equal to number of trees/submodels), to use _as many threads as there are cores on the machine_, and to _process multiple batches in one application run_.
* **In all cases: process as many points as are available in one classify() call.**
