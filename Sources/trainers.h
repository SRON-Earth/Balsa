#ifndef TRAINERS_H
#define TRAINERS_H

#include "traininginterfaces.h"
#include "weightedcoin.h"

/**
 * First working design of the random tree trainer.
 */
class SingleTreeTrainerMark1: public SingleTreeTrainer
{
public:

  SingleTreeTrainerMark1( unsigned int maxDepth = std::numeric_limits<unsigned int>::max()  ):
  SingleTreeTrainer( maxDepth )
  {
  }

  DecisionTree::SharedPointer train( const FeatureIndex &featureIndex, const TrainingDataSet &dataSet );

private:

  WeightedCoin m_coin;

};

#endif // TRAINERS_H
