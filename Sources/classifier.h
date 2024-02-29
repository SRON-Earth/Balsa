#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <memory>

#include "table.h"

/**
 * Abstract interface of a class that can classify data points.
 */
template <typename FeatureIterator, typename OutputIterator>
class Classifier
{
public:

    typedef std::shared_ptr<Classifier>       SharedPointer;
    typedef std::shared_ptr<const Classifier> ConstSharedPointer;

    typedef Table<uint32_t> VoteTable;

    /**
     * Constructor.
     */
    Classifier( std::size_t featureCount ):
    m_featureCount( featureCount )
    {
    }

    /**
     * Destructor.
     */
    virtual ~Classifier()
    {
    }

    /**
     * Returns the number of features used by the classifier.
     */
    std::size_t getFeatureCount() const
    {
        return m_featureCount;
    }

    /**
     * Bulk-classifies a sequence of data points.
     */
    virtual void classify( FeatureIterator pointsStart, FeatureIterator pointsEnd, OutputIterator labels ) const = 0;

    /**
     * Bulk-classifies a set of points, adding a vote (+1) to the vote table for each point of which the label is 'true'.
     * \param pointsStart An iterator that points to the first feature value of the first point.
     * \param pointsEnd An itetartor that points to the end of the block of point data.
     * \param table A table for counting votes.
     * \pre The column count of the vote table must match the number of features, the row count must match the number of points.
     */
    virtual unsigned int classifyAndVote( FeatureIterator pointsStart, FeatureIterator pointsEnd, VoteTable & table ) const = 0;

private:

    std::size_t m_featureCount;
};

#endif // CLASSIFIER_H
