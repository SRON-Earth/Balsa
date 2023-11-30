#ifndef TRAININGINTERFACES_H
#define TRAININGINTERFACES_H

#include <limits>
#include <string>
#include <tuple>
#include <iostream>

#include "decisiontrees.h"

class FeatureIndex;
class TrainingDataSet;

/**
 * Abstract interface of a class that trains a single random decision tree.
 */
class SingleTreeTrainer
{
public:

  SingleTreeTrainer( unsigned int maxDepth = std::numeric_limits<unsigned int>::max()  ):
  m_maxDepth( maxDepth )
  {
      // TODO:
      // Min samples split = 2
      // Min samples leaf = 1
      // Max features = sqrt(nfeatures)
      // Max leaf nodes = None
      // Min impurity decrease = 0.0
      // Bootstrap = True
  }

  virtual ~SingleTreeTrainer()
  {
  }

  /**
   * Train a tree on the provided dataset.
   */
  virtual DecisionTree::SharedPointer train( const FeatureIndex &featureIndex, const TrainingDataSet &dataSet ) = 0;

protected:

  const unsigned int m_maxDepth;

};



#endif // TRAININGINTERFACES_H
