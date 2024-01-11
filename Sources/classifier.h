#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <memory>
#include <valarray>

/**
 * Abstract interface of a class that can classifies data points.
 */
template <typename FeatureIterator, typename OutputIterator>
class Classifier
{
public:

  typedef std::shared_ptr<Classifier> SharedPointer;
  typedef std::shared_ptr<const Classifier> ConstSharedPointer;

  typedef std::valarray<unsigned short> VoteTable;

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
   * Bulk-classifies a set of points, adding a vote (+1) to the vote table for
   * each point of which the label is 'true'.
   */
  virtual unsigned int classifyAndVote( FeatureIterator pointsStart, FeatureIterator pointsEnd, VoteTable & table ) const = 0;

private:

  std::size_t m_featureCount;
};

#endif
