#ifndef RANDOMFORESTCLASSIFIER_H
#define RANDOMFORESTCLASSIFIER_H

#include "decisiontreeclassifierstream.h"
#include "ensembleclassifier.h"

template <typename FeatureIterator,
    typename OutputIterator,
    typename FeatureType = std::iterator_traits<FeatureIterator>::value_type,
    typename LabelType   = std::iterator_traits<OutputIterator>::value_type>
class RandomForestClassifier: public Classifier<FeatureIterator, OutputIterator>
{
public:

    using typename Classifier<FeatureIterator, OutputIterator>::VoteTable;

    RandomForestClassifier( const std::string & modelFileName,
        unsigned int featureCount,
        unsigned int maxThreads = 0,
        unsigned int maxPreload = 1 )
    : Classifier<FeatureIterator, OutputIterator>( featureCount )
    , m_treeStream( modelFileName, maxPreload )
    , m_classifier( featureCount, m_treeStream, maxThreads )
    {
    }

    /**
     * Bulk-classifies a sequence of data points.
     */
    void classify( FeatureIterator pointsStart, FeatureIterator pointsEnd, OutputIterator labels ) const
    {
        m_classifier.classify( pointsStart, pointsEnd, labels );
    }

    /**
     * Bulk-classifies a set of points, adding a vote (+1) to the vote table for
     * each point of which the label is 'true'.
     */
    unsigned int classifyAndVote( FeatureIterator pointsStart, FeatureIterator pointsEnd, VoteTable & table ) const
    {
        return m_classifier.classifyAndVote( pointsStart, pointsEnd, table );
    }

private:

    DecisionTreeClassifierStream<FeatureIterator, OutputIterator, FeatureType, LabelType> m_treeStream;
    EnsembleClassifier<FeatureIterator, OutputIterator> m_classifier;
};

#endif // RANDOMFORESTCLASSIFIER_H
