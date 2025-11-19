---
title: 'Balsa: A Fast C++ Random Forest Classifier with Command-line and Python Interface'
tags:
  - Python
  - C++
  - Random Forest
  - TROPOMI
  - Sentinel 5P
authors:

   - name: Tobias Borsdorff
     corresponding: true
     orcid: 0000-0002-4421-0187
     affiliation: "1"

   - given-names: Denis 
     dropping-particle: de
     surname: Leeuw Duarte
     affiliation: "2"

   - given-names: Joris
     dropping-particle: van
     surname: Zwieten
     affiliation: "2"

   - name: Soumyajit Mandal
     orcid: 0000-0002-2669-4798
     affiliation: "1"

   - name: Jochen Landgraf
     orcid: 0000-0002-6069-0598
     affiliation: "1"

affiliations:
  - name: SRON Space Research Organization Netherlands
    index: 1

  - name: Jigsaw B.V., The Netherlands
    index: 2

date: 10 December 2024
bibliography: paper.bib

---

# Summary

Random Forest classifiers are widely used machine learning methods that combine
multiple decision trees to improve predictive accuracy and reduce overfitting
[@Breiman:2001]. While implementations like scikit-learn [@Pedregosa:2011] are
popular in the Python ecosystem, operational processing environments often
require high-performance C++ implementations that can handle large datasets
efficiently while maintaining low memory footprints.

Balsa is a high-performance, open-source (BSD 3-Clause License) C++
implementation of the Random Forest classifier, designed with runtime
efficiency and memory optimization as core design priorities. The
implementation follows the modern C++17 standard and a complete API
documentation is provided with the package. Originally developed for the
cloud-clearing classification 
in the operational processing of
Copernicus Sentinel-5 Precursor (S5P) methane data [@Lorente:2021;
@Lorente:2023], Balsa addresses the strict performance requirements for
 satellite data processing
[@Borsdorff:2024_paper]. The library has been successfully integrated into ESA's
operational data processing framework [@Borsdorff:2024_atbd;
@Borsdorff:2024_pum], where it currently runs in both the offline and near
real-time S5P methane products, processing large volumes of satellite
observations with stringent latency requirements.

# Statement of Need

Balsa was developed by SRON Netherlands Institute for Space Research in
cooperation with Jigsaw B.V. to meet the demanding performance requirements of
operational satellite data processing. During initial development phases, the
scikit-learn implementation [@Pedregosa:2011] was used, but operational
integration required a C++ implementation with significantly improved runtime
and memory efficiency. The transition to near real-time processing for S5P
methane data further emphasized the need for a solution that could handle
millions of data points with minimal latency and memory overhead. While Balsa
was developed for S5P methane processing, it is designed as a general-purpose
Random Forest classifier applicable to diverse machine learning tasks beyond
satellite data processing.

Balsa offers several key advantages over existing implementations:

- **Performance**: Balsa demonstrates superior runtime performance during the
  training and prediction phase compared to both scikit-learn and the C++-based Ranger
implementation [@Wright:2017]
(\autoref{fig:balsa_runtime}). This advantage is particularly critical for
operational applications where classification speed directly impacts processing
throughput.
- **Memory efficiency**: Balsa consistently shows lower memory footprint during
  both training and prediction phases, making it particularly suitable for
processing large datasets (\autoref{fig:balsa_memory}). Benchmarks demonstrate
scalability to datasets with millions of data points.
- **Accuracy**: All three implementations (Balsa, scikit-learn, and Ranger)
  produce essentially identical prediction accuracy
(\autoref{fig:balsa_accuracy}), ensuring performance improvements stem from
optimization rather than algorithmic compromises.
- **Flexible integration**: Balsa's compact binary format enables seamless
  workflows where models trained in Python can be efficiently loaded and used in
operational C++ environments.
- **Distributed training**: Multiple machines can train Random Forest models
  independently on the same or different datasets, with trained models easily
merged to create stronger classifiers without requiring centralized
coordination.

The performance comparisons presented in \autoref{fig:balsa_runtime},
\autoref{fig:balsa_memory}, and \autoref{fig:balsa_accuracy} were conducted
using the TROPOMI cloud-clearing classification problem as a real-world
benchmark, with datasets derived from TROPOMI satellite measurements as
described in Borsdorff et al. [@Borsdorff:2024_paper].

The library provides three levels of user interaction: a comprehensive C++ API
for direct integration into applications, command-line tools for standalone
training and classification tasks, and Python bindings installable via pip that
simplify development while maintaining access to the high-performance C++ core.
Balsa is cross-platform, supporting Linux, macOS, and Windows environments. This
multi-layered approach supports both rapid prototyping in Python and deployment
in performance-critical production environments. Balsa supports both single-
and double-precision arithmetic, allowing memory optimization as needed. 

# Key Features

Balsa provides a complete ecosystem for Random Forest classification:

**Core Library**: The C++ library supports both binary and multi-class
classification with multithreaded training capabilities. Models can be trained
in parallel across multiple cores and even across multiple independent machines,
with the resulting forests merged to create stronger classifiers. The library
uses an efficient binary format for model storage, enabling fast loading and
minimal disk usage.

**Command-Line Tools**: The package includes utilities for the complete machine
learning workflow: `balsa_generate` creates synthetic datasets for testing,
`balsa_train` trains models with configurable parameters, `balsa_classify`
performs batch classification, `balsa_measure` calculates comprehensive
performance metrics (including accuracy, precision, recall, F-scores, P4 metric,
diagnostic odds ratio, and confusion matrices), `balsa_featureimportance`
analyzes feature contributions following a permutation based method, `balsa_merge` combines independently trained
models for distributed training workflows, and `balsa_test` runs unit tests to
verify installation and functionality.

**Python Interface**: Python bindings provide NumPy integration and a familiar
interface for Python developers, while maintaining the performance benefits of
the underlying C++ implementation. The package is easily installable via pip,
making it readily accessible to the Python machine learning community. Models
trained via Python can be directly used by the C++ tools and vice versa,
facilitating hybrid workflows where development occurs in Python and deployment
in high-performance C++ environments.

**Performance Analysis Tool**: The `rfcperf` benchmarking utility enables
systematic comparison of Random Forest implementations across different dataset
sizes, ranging from thousands to millions of samples. It measures system
performance (CPU time, memory usage, wall-clock time) and classification quality
(accuracy, precision, recall, F-scores) while generating comparative
visualization reports. This tool was used to generate the performance
comparisons presented in \autoref{fig:balsa_runtime},
\autoref{fig:balsa_memory}, and \autoref{fig:balsa_accuracy}, and allows users
to reproduce these benchmarks on their own systems and datasets.

**Comprehensive Documentation**: The package includes detailed documentation
covering installation, theoretical background, optimization guidelines, and
extensive examples for both command-line and programmatic usage.

![Runtime comparison during RFC training (left) and prediction (right) for
scikit-learn (orange), Ranger (green), and Balsa (blue) as a function of dataset
size, evaluated on TROPOMI cloud-clearing data. Balsa demonstrates superior
prediction performance, which is critical for operational applications including
near real-time processing.\label{fig:balsa_runtime}](figures/fig1.png)

![Memory usage during RFC training (left) and prediction (right) for
scikit-learn (orange), Ranger (green), and Balsa (blue) as a function of dataset
size, evaluated on TROPOMI cloud-clearing data. Balsa maintains consistently
lower memory footprint across dataset sizes ranging from thousands to millions
of samples, enabling processing of larger datasets in memory-constrained
environments.\label{fig:balsa_memory}](figures/fig2.png)

![Classification accuracy for scikit-learn (orange), Ranger (green), and Balsa
(blue) as a function of dataset size, evaluated on TROPOMI cloud-clearing data.
All three implementations achieve comparable accuracy, confirming that Balsa's
performance gains do not compromise prediction
quality.\label{fig:balsa_accuracy}](figures/fig3.png)

# Availability
Balsa is publicly available under the BSD 3-Clause License at 
[Balsa GitHub Repository](https://github.com/SRON-Earth/Balsa)


# Authors Contribution
T. Borsdorff led the project and coordinated the overall development. He
contributed to the conceptual design of the software, performed the
verification and validation activities together with J. Landgraf and S. Mandal,
and wrote the manuscript.  J. van Zwieten and D. de Leeuw Duarte were
responsible for the main implementation of the Balsa library, including the
core C++ codebase and associated tools.  All authors contributed to
discussions, refinement of the software, and preparation of the manuscript, and
all agree on the order of authorship.

# Acknowledgements

Balsa development was funded by the European Space Agency.

# References
