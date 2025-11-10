#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <memory>

namespace balsa
{

// Forward declaration.
class ClassifierVisitor;

/**
 * Abstract interface of a class that can classify data points.
 *
 * N.B. by convention, subclasses must provide the following template methods
 * for classification:
 *
 * template<typename FeatureIterator, typename LabelOutputIterator>
 * void classify( FeatureIterator pointsStart, FeatureIterator pointsEnd, LabelOutputIterator labelsStart ) const;
 *
 * template<typename FeatureIterator>
 * unsigned int classifyAndVote( FeatureIterator pointsStart, FeatureIterator pointsEnd, VoteTable & table ) const;
 */
class Classifier
{
public:

    typedef std::shared_ptr<Classifier>       SharedPointer;
    typedef std::shared_ptr<const Classifier> ConstSharedPointer;

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
     * Returns the number of features the classifier expects.
     */
    virtual unsigned int getFeatureCount() const = 0;

    /**
     * Accept a visitor.
     */
    virtual void visit( ClassifierVisitor & visitor ) const = 0;
};

} // namespace balsa

#endif // CLASSIFIER_H
