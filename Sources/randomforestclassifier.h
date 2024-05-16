#ifndef RANDOMFORESTCLASSIFIER_H
#define RANDOMFORESTCLASSIFIER_H

#include "decisiontreeclassifierstream.h"
#include "ensembleclassifier.h"
#include "table.h"

namespace balsa
{

template <typename FeatureIterator = Table<double>::ConstIterator, typename OutputIterator = Table<Label>::Iterator>
class RandomForestClassifier: public Classifier<FeatureIterator, OutputIterator>
{
public:

    using typename Classifier<FeatureIterator, OutputIterator>::VoteTable;

    RandomForestClassifier( const std::string & modelFileName, unsigned int maxThreads = 0, unsigned int maxPreload = 1 ):
    Classifier<FeatureIterator, OutputIterator>(),
    m_classifierStream( modelFileName, maxPreload ),
    m_classifier( m_classifierStream, maxThreads )
    {
    }

    /**
     * Returns the number of classes distinguished by this classifier.
     */
    unsigned int getClassCount() const
    {
        return m_classifier.getClassCount();
    }

    /**
     * Bulk-classifies a sequence of data points.
     */
    void classify( FeatureIterator pointsStart, FeatureIterator pointsEnd, unsigned int featureCount, OutputIterator labelsStart ) const
    {
        m_classifier.classify( pointsStart, pointsEnd, featureCount, labelsStart );
    }

    /**
     * Bulk-classifies a set of points, adding a vote (+1) to the vote table for
     * each point of which the label is 'true'.
     * \param pointsStart An iterator that points to the first feature value of
     *  the first point.
     * \param pointsEnd An itetartor that points to the end of the block of
     *  point data.
     * \param featureCount The number of features for each data point.
     * \param table A table for counting votes.
     * \pre The column count of the vote table must match the number of
     *  features, the row count must match the number of points.
     */
    unsigned int classifyAndVote( FeatureIterator pointsStart, FeatureIterator pointsEnd, unsigned int featureCount, VoteTable & table ) const
    {
        return m_classifier.classifyAndVote( pointsStart, pointsEnd, featureCount, table );
    }

    void setClassWeights( const std::vector<float> &classWeights )
    {
        m_classifier.setClassWeights( classWeights );
    }

private:

    DecisionTreeClassifierStream<FeatureIterator, OutputIterator> m_classifierStream;
    EnsembleClassifier<FeatureIterator, OutputIterator>           m_classifier;
};

} // namespace balsa

#endif // RANDOMFORESTCLASSIFIER_H
