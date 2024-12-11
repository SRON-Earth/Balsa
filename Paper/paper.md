---
title: 'Balsa: A Fast C++ Random Forest Classifier with Commandline and Python Interface'
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
     equal-contribution: true
     affiliation: "1"

   - given-names: Denis 
     dropping-particle: de
     surname: Leeuw Duarte
     equal-contribution: true
     affiliation: "2"

   - given-names: Joris
     dropping-particle: van
     surname: Zwieten
     equal-contribution: true
     affiliation: "2"

   - name: Soumyajit Mandal
     corresponding: true
     orcid: 0000-0002-2669-4798
     equal-contribution: true
     affiliation: "1"

   - name: Jochen Landgraf
     orcid: 0000-0002-6069-0598
     equal-contribution: true
     affiliation: "1"


#  - name: Adrian M. Price-Whelan
#    orcid: 0000-0000-0000-0000
#    equal-contrib: true
#    affiliation: "1, 2" # (Multiple affiliations must be quoted)
#  - name: Author Without ORCID
#    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
#    affiliation: 2
#  - name: Author with no affiliation
#    corresponding: true # (This is how to denote the corresponding author)
#    affiliation: 3
#  - given-names: Ludwig
#    dropping-particle: van
#    surname: Beethoven
#    affiliation: 3
affiliations:
  - name: SRON Netherlands Institute for Space Research, The Netherlands
    index: 1

  - name: Jigsaw B.V., The Netherlands
    index: 2

# - name: Lyman Spitzer, Jr. Fellow, Princeton University, United States
#   index: 1
#   ror: 00hx57361
# - name: Institution Name, Country
#   index: 2
# - name: Independent Researcher, Country
#   index: 3
date: 10 Debember 2024
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
#aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
#aas-journal: Astrophysical Journal <- The name of the AAS journal.

---

# Summary

A random forest classifier is a widely used machine learning method that builds
upon the strengths of decision trees [@Pedregosa:2011]. It combines the outputs
of multiple decision trees to improve predictive accuracy, reduce the risk of
overfitting, and effectively identify outliers within datasets. A decision tree
is a simple, intuitive model that splits input data into subsets based on the
values of specific features, forming a hierarchical structure with decision
nodes leading to predicted outcomes [@Breiman:2001].  Balsa is a highly
efficient C++ implementation of the random forest classifier concept, built
with both runtime and memory performance as key design priorities. Balsa
provides multi-threaded and distributed training capabilities, allowing users
to scale machine learning processes across multiple computing cores or
distributed systems by training separate random forests and combining them at
the end. It supports both binary classification tasks as well as multi-label
classification, expanding its use for a variety of machine learning challenges.
Moreover, Balsa includes comprehensive tools for feature importance evaluation,
prediction performance metrics, and statistical analysis, enabling users to
gain insights into their data and model performance.  To ensure ease of use,
Balsa employs a compact and storage-efficient binary format for saving fully
trained random forests. This allows models to be quickly reloaded for future
predictions. Designed to fit seamlessly into existing C++ development
workflows, Balsa can integrate easily into custom machine learning pipelines.
Alternatively, users can employ Balsa through the command line interface or via
a versatile Python interface, which can be easily installed with pip. This
design provides flexibility for users working in diverse programming
environments and across a wide range of machine learning use cases.  Balsa
performance, flexibility, and ease of integration make it a reliable choice for
researchers and developers who require a fast, scalable, and efficient random
forest implementation for a variety of machine learning tasks.  The software
package is hosted on GitHub, accompanied by a comprehensive user guide. This
guide includes detailed installation instructions and multiple examples
demonstrating various use cases [@balsagit].

# Statement of need

Balsa was developed by SRON and Jigsaw to support ESA's operational processing
of TROPOMI CH4 satellite data [@Lorente:2021; @Lorente:2023], specifically to
identify and remove measurements contaminated by cloud interference
[@Borsdorff:2024_paper]. During the beta phase , the team utilized the
excellent scikit-learn (sklearn) implementation [@Pedregosa:2011] of the random
forest classifier concept . However, for integration into an operational
framework, a C++ implementation with improved runtime and memory efficiency was
required.  This led to the development of Balsa, which is designed to mimic the
sklearn implementation while addressing these performance demands.  Balsa is
now fully operational within ESA's data processing framework
[@Borsdorff:2024_atbd; @Borsdorff:2024_pum]. Further planned applications
include improving the posteriori quality filtering of TROPOMI data products.
One key challenge involves managing multiple random forests simultaneously in
memory to optimize efficiency. Moreover, Balsa will play a pivotal role in
supporting the upcoming Near Real-Time TROPOMI CH4 product, which requires fast
and efficient predictions to meet strict processing time constraints. Although
Balsa was developed in support of TROPOMI CH4 processing, it is designed to be
entirely independent of any specific application. It serves as a universal
machine learning toolbox that can be applied across a variety of use cases
beyond TROPOMI data processing. Its flexibility, high performance, and ease of
integration make it an invaluable tool for any application requiring scalable,
efficient random forest-based machine learning.

# Acknowledgements

Balsa was developed for the Netherlands Institute of Space Research by Jigsaw
B.V. in The Netherlands, using funding from the European Space Agency.

# References
