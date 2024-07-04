#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <memory>

#include "table.h"

namespace balsa
{


// Forward declarations of all supported classifiers.
class EnsembleClassifier;
template<typename FeatureType> class DecisionTreeClassifier;

/**
 * Base class for visiting Classifiers.
 */
class ClassifierVisitor
{
public:

  virtual ~ClassifierVisitor();
  virtual void visit( const EnsembleClassifier             &classifier );
  virtual void visit( const DecisionTreeClassifier<float>  &classifier );
  virtual void visit( const DecisionTreeClassifier<double> &classifier );

};

/**
 * Abstract interface of a class that can classify data points.
 * N.B. by convention, subclasses must provide template methods for classification:
 *
 * template<typename FeatureIterator, typename LabelOutputIterator>
 * void classify( FeatureIterator pointsStart, FeatureIterator pointsEnd, LabelOutputIterator labelsStart ) const;
 * unsigned int classifyAndVote( FeatureIterator pointsStart, FeatureIterator pointsEnd, VoteTable & table ) const;
 *
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
     * Accept a visitor.
     */
    virtual void visit( ClassifierVisitor & visitor ) = 0;
};

} // namespace balsa

#endif // CLASSIFIER_H
