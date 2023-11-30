#ifndef INGESTION_H
#define INGESTION_H

#include <string>

#include "datarepresentation.h"

/**
 * Load a data set from file.
 */
DataSet::SharedPointer loadDataSet(const std::string &dataFile);

/**
 * Load a labelled data set from file.
 */
TrainingDataSet::SharedPointer loadTrainingDataSet(const std::string &dataFile, const std::string &labelFile);

#endif // INGESION_H
