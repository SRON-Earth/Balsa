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

A Random Forest classifier is a widely used machine learning method. It builds on the strengths of decision trees [@Pedregosa:2011] which are simple, intuitive models to classify input data into subsets based on the values of specific features. They form hierarchical structures with decision nodes leading to prediction [@Breiman:2001]. Combining the output of multiple decision trees improves its predictive accuracy, reduces the risk of overfitting, and effectively identifies outliers within data sets. 
 Balsa is a highly efficient C++ implementation of the Random Forest classifier concept, built with a focus on runtime and memory performance as key design priorities. Balsa provides multithreaded and distributed training capabilities, allowing users to scale machine learning processes across multiple computing cores or distributed systems by training separate Random Forests and combining them at the end. It supports both binary classification tasks and multi-label classification, expanding its use for a variety of machine learning challenges. Furthermore, Balsa includes comprehensive tools to assess the importance of features, prediction performance metrics, and statistical analysis, allowing users to gain insight into their data and model performance. To ensure ease of use, Balsa uses a compact and storage-efficient binary format to save trained Random Forests. This allows models to be quickly reloaded for future predictions. Designed to fit seamlessly into existing C++ development workflows, Balsa can be easily integrated into custom machine learning pipelines. Alternatively, users can employ Balsa through a command line interface or via a versatile Python interface, the latter can be easily installed with pip. This design provides flexibility for users working in diverse programming environments and across a wide range of machine learning use cases. Balsa performance, flexibility, and ease of integration make it a reliable choice for researchers and developers who require fast, scalable, and efficient Random Forest implementation for a variety of machine-learning tasks. The Balsa package is an open-source software hosted on GitHub and is accompanied by a comprehensive user guide. This guide includes detailed installation instructions and multiple examples that demonstrate various use cases [@balsagit].

# Statement of need
Balsa has been developed by SRON Netherlands Institute for Space Research and Jigsaw B.V., The Netherlands,in support of the operational processing of Copernicus Sentinel-5 Precursor (S5P) methane data [@Lorente:2021; @Lorente:2023]. The processing requires strict clearing of the measurement data regarding cloud interference, leading to a classification problem [@Borsdorff:2024_paper]. During the beta phase of software development, we utilized the scikit-learn (sklearn) implementation [@Pedregosa:2011] of the Random Forest classifier concept. Integrating into an operational processing framework required a C++ implementation with improved run-time and memory efficiency. This led to the development of Balsa, which is designed to build on the sklearn implementation, addressing the performance demands of the satellite mission. Balsa is fully operational within ESA's data processing framework [@Borsdorff:2024_atbd; @Borsdorff:2024_pum]. Future applications of Blase will focus on the key challenge of managing multiple Random Forests simultaneously in memory to optimize efficiency. In addition, Balsa will play a crucial role in supporting the upcoming near real-time S5P methane product, which requires fast and efficient data classification to meet strict processing time constraints. Although Balsa was developed for S5P methane processing, it is designed to be a Random Forest classifier independent of any specific application. It serves as a universal machine learning toolbox that can be applied in a variety of use cases beyond the S5P data processing. Its flexibility, high performance, and ease of integration make it an invaluable tool for any application that requires efficient and scalable Random Forest-based machine learning.


# Acknowledgements

Balsa development was funded by the European Space Agency.

# References
