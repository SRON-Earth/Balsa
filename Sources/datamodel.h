#ifndef DATAMODEL_H
#define DATAMODEL_H

#include <vector>

/**
 * One data point in a data set. The data consists of a list of feature-values,
 * where each feature is a double-precision float.
 */
typedef std::vector<double> DataPoint;

/**
 * A label (e.g. 'cloud'/'no cloud' ) for a classification problem.
 * By convention, labels are represented as small unsigned integers.
 */
typedef uint8_t DataPointLabel;

/**
 * A set of DataPoints.
 */
class DataSet
{
public:

  /**
   * The unique consecutive ID of a DataPoint.
   */
  typedef std::size_t DataPointID;

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

  TrainingDataSet( unsigned int featureCount ):
  m_dataSet( featureCount )
  {
  }

  /**
   * Append a data point and its known label to the set.
   * \pre The number of features in the point must match those in this dataset (dataPoint.size() == this->getFeatureCount()).
   * \return The unique consecutive ID of the point.
   */
  DataSet::DataPointID appendDataPoint( const DataPoint &dataPoint, DataPointLabel label )
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
  DataPointLabel getLabel( DataSet::DataPointID pointID ) const
  {
      return m_dataSetLabels[ pointID ];
  }

  /**
   * Returns a specific feature-value of a particular point.
   */
  double getFeatureValue( DataSet::DataPointID pointID, unsigned int featureID ) const
  {
      return m_dataSet.getFeatureValue( pointID, featureID );
  }

  void dump()
  {
      // Print all point IDs, features, and labels.
      auto featureCount = m_dataSet.getFeatureCount();
      for ( DataSet::DataPointID pointID = 0; pointID < m_dataSet.size(); ++pointID )
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

  DataSet                       m_dataSet      ; // The data points without their labels.
  std::vector< DataPointLabel > m_dataSetLabels; // The labels of each point in the dataset.
};

/**
 * An ordered (sorted) view on a TrainingDataSet, that allows fast, ordered traversal through the points.
 */
class FeatureIndex
{
public:

  /**
   * Each entry in the FeatureIndex is a tuple of the feature value of a point, its known label, and its unique ID.
   */
  typedef std::tuple< double, DataPointLabel, DataSet::DataPointID > Entry;

  /**
   * Creates a FeatureIndex for the specified TrainingDataSet.
   * N.B. The index will not cover points that are appended to the dataset after this index is created.
   */
  FeatureIndex( const TrainingDataSet &dataset )
  {
      // Create a sorted index for each feature.
      m_featureIndices.reserve( dataset.getFeatureCount() );
      for ( unsigned int feature = 0; feature < dataset.getFeatureCount(); ++feature )
      {
          // Create the index for this feature, and give it enough capacity.
          m_featureIndices.push_back( std::vector< Entry >() );
          auto &index = *m_featureIndices.rbegin();
          index.reserve( dataset.size() );

          // Create entries for each point in the dataset.
          for ( DataSet::DataPointID pointID( 0 ), end( dataset.size() ); pointID < end; ++pointID )
          {
              index.push_back( Entry( dataset.getFeatureValue( pointID, feature ), dataset.getLabel( pointID ), pointID ) );
          }

          // Sort the index by feature value (the rest of the fields don't matter..
          std::sort( index.begin(), index.end() );
      }
  }

private:

  std::vector< std::vector< Entry > > m_featureIndices;

};

#endif // DATAMODEL_H
