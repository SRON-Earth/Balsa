#ifndef TRAINER_MARK_2_H
#define TRAINER_MARK_2_H

#include "traininginterfaces.h"

/**
 * Random tree trainer that represents the tree as a single array.
 */
class SingleTreeTrainerMark2: public SingleTreeTrainer
{
public:

  SingleTreeTrainerMark2( unsigned int maxDepth = std::numeric_limits<unsigned int>::max()  ):
  SingleTreeTrainer( maxDepth )
  {
  }

  DecisionTree::SharedPointer train( const FeatureIndex &featureIndex, const TrainingDataSet &dataSet );

};

#endif // TRAINERS_H
