#ifndef TRAINERS_H
#define TRAINERS_H

#include "traininginterfaces.h"

/**
 * Random tree trainer that represents the tree as a single array.
 */
template <typename FeatureValueType = double, typename LabelValueType = bool>
class SingleTreeTrainerMark2: public SingleTreeTrainer<FeatureValueType, LabelValueType>
{
public:

    SingleTreeTrainerMark2( unsigned int maxDepth = std::numeric_limits<unsigned int>::max() )
    : SingleTreeTrainer<FeatureValueType, LabelValueType>( maxDepth )
    {
    }

    typename DecisionTree<FeatureValueType, LabelValueType>::SharedPointer train( const FeatureIndex & featureIndex,
        const TrainingDataSet & dataSet, unsigned int featuresToScan, bool writeGraphviz, unsigned int treeID );
};

template <>
DecisionTree<>::SharedPointer SingleTreeTrainerMark2<>::train( const FeatureIndex & featureIndex,
    const TrainingDataSet & dataSet, unsigned int featuresToScan, bool writeGraphviz, unsigned int treeID );


#endif // TRAINERS_H
