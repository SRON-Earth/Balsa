#ifndef DATAMODEL_H
#define DATAMODEL_H

#include <algorithm>
#include <cassert>
#include <limits>
#include <memory>
#include <vector>

/**
 * One data point in a data set. The data consists of a list of feature-values,
 * where each feature is a double-precision float.
 */
typedef std::vector<double> DataPoint;

/**
 * The unique consecutive ID of a DataPoint.
 */
typedef std::size_t DataPointID;


/**
 * A set of DataPoints.
 */
class DataSet
{
public:

  DataSet( unsigned int featureCount ):
  m_featureCount( featureCount )
  {
  }

  /**
   * Append a data point to the set.
   * \pre The number of features in the point must match those in this dataset (dataPoint.size() == this->getFeatureCount()).
   * \return The unique consecutive ID of the point.
   */
  DataPointID appendDataPoint( const DataPoint &dataPoint )
  {
      // Check precondition, append the features of the datapoint to the end of the data array.
      assert( dataPoint.size() == getFeatureCount() );
      std::copy( dataPoint.begin(), dataPoint.end(), std::back_inserter( m_dataRows ) );
      return ( m_dataRows.size() / m_featureCount ) - 1;
  }

  /**
   * Returns the number of features in all datapoints in this dataset.
   */
  unsigned int getFeatureCount() const
  {
      return m_featureCount;
  }

  /**
   * Returns the number of data points in this dataset.
   */
  std::size_t size() const
  {
      return m_dataRows.size() / m_featureCount;
  }

  /**
   * Returns a specific feature-value of a particular point.
   */
  double getFeatureValue( DataPointID pointID, unsigned int featureID ) const
  {
      assert( pointID < size() );
      assert( featureID < m_featureCount );
      return m_dataRows[ pointID * m_featureCount + featureID];
  }

private:

  unsigned int        m_featureCount;
  std::vector<double> m_dataRows    ; // Data is stored in row-major order, one row of size m_featureCount per row;

};

/**
 * A set of DataPoints that includes the known labels of each point.
 */
class TrainingDataSet
{
public:

  typedef std::shared_ptr<const TrainingDataSet> ConstSharedPointer;
  typedef std::shared_ptr<      TrainingDataSet> SharedPointer     ;

  TrainingDataSet( unsigned int featureCount ):
  m_dataSet( featureCount )
  {
  }

  /**
   * Append a data point and its known label to the set.
   * \pre The number of features in the point must match those in this dataset (dataPoint.size() == this->getFeatureCount()).
   * \return The unique consecutive ID of the point.
   */
  DataPointID appendDataPoint( const DataPoint &dataPoint, bool label )
  {
      // Add the datapoint to the dataset.
      auto id = m_dataSet.appendDataPoint( dataPoint );

      // Add the label to the label set.
      m_dataSetLabels.push_back( label );
      assert( m_dataSetLabels.size() == m_dataSet.size() );

      // Return the ID of the created point.
      return id;
  }

  /**
   * Returns the number of features in each point.
   */
  std::size_t getFeatureCount() const
  {
      return m_dataSet.getFeatureCount();
  }

  /**
   * Returns the number of points in the training data set.
   */
  std::size_t size() const
  {
      return m_dataSet.size();
  }

  /**
   * Returns the known label of a point.
   */
  bool getLabel( DataPointID pointID ) const
  {
      return m_dataSetLabels[ pointID ];
  }

  /**
   * Returns a specific feature-value of a particular point.
   */
  double getFeatureValue( DataPointID pointID, unsigned int featureID ) const
  {
      return m_dataSet.getFeatureValue( pointID, featureID );
  }

  void dump()
  {
      // Print all point IDs, features, and labels.
      auto featureCount = m_dataSet.getFeatureCount();
      for ( DataPointID pointID = 0; pointID < m_dataSet.size(); ++pointID )
      {
          std::cout << pointID;
          for ( unsigned int feature = 0; feature < featureCount; ++feature )
          {
              std::cout << ", " << m_dataSet.getFeatureValue( pointID, feature );
          }
          std::cout << ", " << static_cast<unsigned int>( m_dataSetLabels[pointID] ) << std::endl;
      }
  }

private:

  DataSet             m_dataSet      ; // The data points without their labels.
  std::vector< bool > m_dataSetLabels; // The labels of each point in the dataset.
};


/**
 * An index for traversing points in a dataset in order of each feature.
 */
class FeatureIndex
{
  typedef std::tuple< double, bool, DataPointID > Entry;

public:

  FeatureIndex( const TrainingDataSet &dataset )
  {
      // Create a sorted index for each feature.
      m_featureIndices.clear();
      m_featureIndices.reserve( dataset.getFeatureCount() );
      for ( unsigned int feature = 0; feature < dataset.getFeatureCount(); ++feature )
      {
          // Create the index for this feature, and give it enough capacity.
          m_featureIndices.push_back( std::vector< Entry >() );
          auto &index = *m_featureIndices.rbegin();
          index.reserve( dataset.size() );

          // Create entries for each point in the dataset.
          for ( DataPointID pointID( 0 ), end( dataset.size() ); pointID < end; ++pointID )
          {
              index.push_back( Entry( dataset.getFeatureValue( pointID, feature ), dataset.getLabel( pointID ), pointID ) );
          }

          // Sort the index by feature value (the rest of the fields don't matter..
          std::sort( index.begin(), index.end() );
      }
  }

private:

  std::vector< std::vector< Entry > >  m_featureIndices;
};

class SingleTreeTrainer
{
public:
  SingleTreeTrainer( const FeatureIndex &featureIndex ):
  m_featureIndex( featureIndex )
  {
  }

private:

  const FeatureIndex &m_featureIndex;
};

/**
 * Trains a random binary forest classifier on a TrainingDataSet.
 */
class BinaryRandomForestTrainer
{
public:

  /**
   * Constructor.
   * \param dataset A const reference to a training dataset. Modifying the set after construction of the trainer invalidates the trainer.
   * \param concurrentTrainers The maximum number of trees that may be trained concurrently.
   */
  BinaryRandomForestTrainer( TrainingDataSet::ConstSharedPointer dataset, unsigned int concurrentTrainers = 10 ):
  m_dataSet( dataset ),
  m_featureIndex( *dataset ),
  m_trainerCount( concurrentTrainers )
  {
  }

  /**
   * Destructor.
   */
  virtual ~BinaryRandomForestTrainer()
  {
  }

  /**
   * Train a forest of random trees on the data.
   */
  void train()
  {
      // Create the specified number of (possibly) concurrent single-tree trainers.
      m_treeTrainers.clear();
      for ( unsigned int i = 0; i < m_trainerCount; ++i )
          m_treeTrainers.push_back( SingleTreeTrainer( m_featureIndex ) );
  }

private:

  TrainingDataSet::ConstSharedPointer  m_dataSet     ;
  FeatureIndex                         m_featureIndex;
  unsigned int                         m_trainerCount;
  std::vector< SingleTreeTrainer    >  m_treeTrainers;

};

/**
 * Compute the Gini impurity of a set of totalCount points, where trueCount points are labeled 'true', and the rest is false.
 */
inline double giniImpurity( unsigned int trueCount, unsigned int totalCount )
{
    double t = trueCount;
    auto T   = totalCount;
    return ( 2 * t * ( 1.0 - ( t / T ) ) ) / T;
}

class TrainingTreeNode
{
public:

  typedef std::shared_ptr<TrainingTreeNode> SharedPointer;

  TrainingTreeNode( unsigned int totalPointCount, unsigned int trueCount ):
  m_totalCount( totalPointCount ),
  m_trueCount( trueCount )
  {
  }

  /**
   * Initializes the search for the optimal split.
   */
  void initializeOptimalSplitSearch()
  {
      // Reset the best split found so far.
      m_bestSplitFeature   = 0;
      m_bestSplitValue     = std::numeric_limits<double>::min();
      m_bestSplitGiniIndex = std::numeric_limits<double>::max();
  }

  /**
   * Instructs this node and its children that a particular feature will be traversed in-order now.
   */
  void startFeatureTraversal( unsigned int featureID )
  {
      // Start feature traversal in the children, if present.
      if ( m_leftChild )
      {
          assert( m_rightChild );
          m_leftChild->startFeatureTraversal( featureID );
          m_rightChild->startFeatureTraversal( featureID );
      }

      // Register which feature is being traversed now, and reset traversal statistics.
      m_currentFeature     = featureID;
      m_trueCountLeftHalf  = 0;
      m_trueCountRightHalf = m_trueCount;
  }

  /**
   * Visit one point during the feature traversal phase.
   */
  void visitPoint( DataPointID, double featureValue, bool label )
  {
      // Update the traversal statistics.
      if ( label )
      {
          ++m_trueCountLeftHalf;
          --m_trueCountRightHalf;
      }
      ++m_totalCountLeftHalf;

      // Compute the Gini index, assuming the split is made at this point.
      auto totalCountRightHalf = m_totalCount - m_totalCountLeftHalf;
      auto trueCountRightHalf  = m_trueCount - m_trueCountLeftHalf;
      auto giniLeft  = giniImpurity( m_trueCountLeftHalf, m_totalCountLeftHalf );
      auto giniRight = giniImpurity( trueCountRightHalf , totalCountRightHalf  );
      auto giniTotal = ( giniLeft * m_totalCountLeftHalf + giniRight * totalCountRightHalf ) / m_totalCount;

      // Save this split if it is the best one so far.
      if ( giniTotal < m_bestSplitGiniIndex )
      {
          m_bestSplitFeature   = m_currentFeature;
          m_bestSplitValue     = featureValue    ;
          m_bestSplitGiniIndex = giniTotal       ;
      }
  }

  SharedPointer m_leftChild         ;
  SharedPointer m_rightChild        ;
  unsigned int  m_splitFeatureID    ;
  double        m_splitFeature      ;
  bool          m_label             ; // Only valid for nodes that don't have children.
  unsigned int  m_totalCount        ;
  unsigned int  m_trueCount         ;
  unsigned int  m_totalCountLeftHalf;
  unsigned int  m_trueCountLeftHalf ;
  unsigned int  m_trueCountRightHalf;
  unsigned int  m_currentFeature    ; // The feature that is currently being traversed.

  unsigned int m_bestSplitFeature  ;
  double       m_bestSplitValue    ;
  double       m_bestSplitGiniIndex;

};

#endif // DATAMODEL_H
