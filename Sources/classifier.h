#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <memory>

#include "table.h"

namespace balsa
{

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
     * Destructor.
     */
    virtual ~Classifier()
    {
    }

    /**
     * Returns the number of classes distinguished by this classifier.
     */
    virtual unsigned int getClassCount() const = 0;

    /**
     * Bulk-classifies a sequence of data points.
     */
    virtual void classify( FeatureIterator pointsStart, FeatureIterator pointsEnd, unsigned int featureCount, OutputIterator labelsStart ) const = 0;

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
     virtual unsigned int classifyAndVote( FeatureIterator pointsStart, FeatureIterator pointsEnd, unsigned int featureCount, VoteTable & table ) const = 0;
};

} // namespace balsa

#endif // CLASSIFIER_H
