#ifndef DATAMODEL_H
#define DATAMODEL_H

#include <iostream>
#include <fstream>
#include <random>
#include <iomanip>
#include <algorithm>
#include <cassert>
#include <limits>
#include <memory>
#include <vector>
#include <thread>
#include <semaphore>

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
 * Compute the Gini impurity of a set of totalCount points, where trueCount points are labeled 'true', and the rest is false.
 */
inline double giniImpurity( unsigned int trueCount, unsigned int totalCount )
{
    double t = trueCount;
    auto T   = totalCount;
    return ( 2 * t * ( 1.0 - ( t / T ) ) ) / T;
}

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

  void dump() const
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

// Forward declarations.
class DecisionTreeNode;
class DecisionTreeInternalNode;
class DecisionTreeLeafNode;

/**
 * A Visitor for decision trees.
 */
class DecisionTreeVisitor
{
public:
  virtual void accept( const DecisionTreeInternalNode & ) {}
  virtual void accept( const DecisionTreeLeafNode     & ) {}
};

/**
 * A node in a decision tree. N.B. this class is intended for evaluation purposes, not for training (see TrainingTreeNode).
 */
class DecisionTreeNode
{
public:

  typedef std::shared_ptr <      DecisionTreeNode> SharedPointer     ;
  typedef std::shared_ptr <const DecisionTreeNode> ConstSharedPointer;

  DecisionTreeNode()
  {
  }

  virtual ~DecisionTreeNode()
  {
  }

  /**
   * Classify all points in a DataSet.
   */
  std::vector<bool> classify( const DataSet &dataSet ) // TODO: Make this an interator-based, container independent template.
  {
      std::vector< bool > labels(dataSet.size());
      for (std::size_t i = 0; i < dataSet.size(); ++i)
      {
        labels.at(i) = classify(dataSet, i);
      }
      return labels;
  }

  /**
   * Return the classification of all data points in a data set.
   * N.B. This is a naive implementation, suitable for testing and low-performance applications.
   */
  virtual bool classify( const DataSet &dataSet, DataPointID pointID ) const = 0;

  /**
   * Return the classification of a data point.
   * N.B. This is a naive implementation, suitable for testing and low-performance applications.
   */
  virtual bool classify( const DataPoint &point ) const = 0; // TODO: add efficient bulk classifier.

  /**
   * Returns the number of nodes in this tree.
   */
  virtual unsigned int getNodeCount() const = 0;

  /**
   * Print routine for testing purposes.
   */
  virtual void dump( unsigned int indent = 0 ) const = 0;

  /**
   * Visitor pattern implementation.
   */
  virtual void visit( DecisionTreeVisitor &visitor ) = 0;

   /**
   * Visitor pattern implementation.
   */
  virtual void visit( DecisionTreeVisitor &visitor ) const = 0;

};

/**
 * An internal node in a decision tree.
 */
class DecisionTreeInternalNode: public DecisionTreeNode
{
public:

  /**
   * Constructor.
   * \param featureID The feature on which the node splits the dataset.
   * \param splitValue The value on which the node splits the dataset (x < splitValue is left half, x >= is right half).
   */
  DecisionTreeInternalNode( unsigned int featureID, double splitValue, DecisionTreeNode::SharedPointer leftChild, DecisionTreeNode::SharedPointer rightChild ):
  m_featureID ( featureID  ),
  m_splitValue( splitValue ),
  m_leftChild ( leftChild  ),
  m_rightChild( rightChild )
  {
  }

  /**
   * Visitor pattern implementation.
   */
  virtual void visit( DecisionTreeVisitor &visitor )
  {
      visitor.accept( *this );
  }

  /**
   * Visitor pattern implementation.
   */
  virtual void visit( DecisionTreeVisitor &visitor ) const
  {
      visitor.accept( *this );
  }

  /**
   * Returns the left child of this node.
   */
  DecisionTreeNode::SharedPointer getLeftChild() const
  {
      return m_leftChild;
  }

  /**
   * Returns the right child of this node.
   */
  DecisionTreeNode::SharedPointer getRightChild() const
  {
      return m_rightChild;
  }

  /**
   * Returns the ID of the feature at which this node splits.
   */
  unsigned int getSplitFeature() const
  {
      return m_featureID;
  }

  /**
   * Returns the value at which this node splits.
   */
  double getSplitValue() const
  {
      return m_splitValue;
  }

  /**
   * Implementation of base class method.
   */
  bool classify( const DataSet &dataSet, DataPointID pointID ) const
  {
      if ( dataSet.getFeatureValue( pointID, m_featureID ) < m_splitValue )
          return m_leftChild->classify( dataSet, pointID );
      return m_rightChild->classify( dataSet, pointID );
  }

  /**
   * Implementation of base class method.
   */
  bool classify( const DataPoint &point ) const
  {
      if ( point[m_featureID] < m_splitValue ) return m_leftChild->classify( point );
      return m_rightChild->classify( point );
  }

  unsigned int getNodeCount() const
  {
      return m_leftChild->getNodeCount() + m_rightChild->getNodeCount();
  }

  virtual void dump( unsigned int indent ) const
  {
      auto tab = std::string( indent, ' ' );
      std::cout << tab << "Feature #" << m_featureID << ", split value = " << std::setprecision( 17 ) <<  m_splitValue << std::endl;
      std::cout << tab << "Left:" << std::endl;
      m_leftChild->dump( indent + 1 );
      std::cout << tab << "Right:" << std::endl;
      m_rightChild->dump( indent + 1 );
  }

private:

  unsigned int                    m_featureID ;
  double                          m_splitValue;
  DecisionTreeNode::SharedPointer m_leftChild ;
  DecisionTreeNode::SharedPointer m_rightChild;

};

/**
 * Leaf node in a binary decision tree.
 */
class DecisionTreeLeafNode: public DecisionTreeNode
{
public:

  /**
   * Constructor.
   */
  DecisionTreeLeafNode( bool label ):
  m_label( label )
  {
  }

  /**
   * Visitor pattern implementation.
   */
  virtual void visit( DecisionTreeVisitor &visitor )
  {
      visitor.accept( *this );
  }

  /**
   * Visitor pattern implementation.
   */
  virtual void visit( DecisionTreeVisitor &visitor ) const
  {
      visitor.accept( *this );
  }

  /**
   * Implementation of base class method.
   */
  bool classify( const DataSet &, DataPointID ) const
  {
      return getLabel();
  }

  /**
   * Implementation of base class method.
   */
  bool classify( const DataPoint & ) const
  {
      return getLabel();
  }

  unsigned int getNodeCount() const
  {
      return 1;
  }

  /**
   * Returns the label of (all points in) this node.
   */
  bool getLabel() const
  {
      return m_label;
  }

  /**
   * Implementation of base class method.
   */
  virtual void dump( unsigned int indent ) const
  {
      auto tab = std::string( indent, ' ' );
      std::cout << tab << ( m_label ? "TRUE" : "FALSE" ) << std::endl;
  }

private:

  bool m_label;

};

/**
 * Saves decision trees to a file.
 */
class DecisionTreeWriter: public DecisionTreeVisitor
{
public:

  /**
   * Creates a writer that writes to the specified file.
   * \param binaryFile A writable, open, binary-mode stream.
   */
  DecisionTreeWriter( std::ofstream &binaryFile ):
  m_file( binaryFile )
  {
  }

  void accept( const DecisionTreeInternalNode &node )
  {
      // Wrile the 'internal node' marker, the members, and the children.
      m_file << 'i' << static_cast<unsigned char>( node.getSplitFeature() );
      auto value = node.getSplitValue();
      m_file.write( reinterpret_cast<const char *>( &value ), sizeof( value ) );
      node.getLeftChild ()->visit( *this );
      node.getRightChild()->visit( *this );
  }

  void accept( const DecisionTreeLeafNode &node )
  {
      // Write the 'leaf node' marker and the label.
      m_file << 'l' << ( node.getLabel() ? 'T' : 'F' );
  }

private:

  std::ofstream &m_file;
};

/**
 * Writes a decision tree to a binary file.
 */
inline void writeToFile( const DecisionTreeNode &tree, const std::string &filename )
{
    // Serialize the tree.
    std::ofstream out( filename, std::ios::binary | std::ios::out );
    DecisionTreeWriter writer( out );
    const_cast<DecisionTreeNode &>( tree ).visit( writer );
    out.close();
};

/**
 * An enseble of (usually randomized) decision trees.
 */
class Forest
{
public:

  typedef std::shared_ptr<Forest> SharedPointer;
  typedef std::vector<DecisionTreeNode::ConstSharedPointer>::const_iterator ConstIterator;

  ConstIterator begin() const
  {
      return m_trees.begin();
  }

  ConstIterator end() const
  {
      return m_trees.end();
  }

  /**
   * Classify all points in a DataSet.
   */
  std::vector<bool> classify( const DataSet &dataSet ) // TODO: Make this an interator-based, container independent template.
  {
      std::vector<bool> labels( dataSet.size() );
      for ( std::size_t i = 0; i < dataSet.size(); ++i )
      {
        labels[i] = classify( dataSet, i );
      }
      return labels;
  }

  /**
   * Return the classification of all data points in a data set.
   */
  bool classify( const DataSet &dataSet, DataPointID pointID )
  {
      unsigned int trueCount = 0;
      for ( auto tree: m_trees ) if ( tree->classify( dataSet, pointID ) ) ++trueCount;
      return trueCount >= m_trees.size() / 2; // TODO: weighted voting based on quality of out-of-bag predictions of subtrees?
  }

  /**
   * Return the classification of a data point.
   */
  bool classify( const DataPoint &point ) const
  {
      unsigned int trueCount = 0;
      for ( auto tree: m_trees ) if ( tree->classify( point ) ) ++trueCount;
      return trueCount >= m_trees.size() / 2; // TODO: weighted voting based on quality of out-of-bag predictions of subtrees?
  }

  /**
   * Add a tree to the forest.
   * \param tree The root node of the tree.
   */
  void addTree( DecisionTreeNode::ConstSharedPointer tree )
  {
      m_trees.push_back( tree );
  }

  /**
   * Print the forest for debugging purposes.
   */
  void dump() const
  {
      std::cout << "Forest:" << std::endl;
      unsigned int i = 0;
      std::string rule( 80, '-' );
      for ( auto &tree: m_trees )
      {
          // Dump the tree with a header.
          std::cout << rule << std::endl;
          std::cout << "Tree #" << i << std::endl;
          std::cout << rule << std::endl;
          tree->dump();

          // Increment counter.
          ++i;
      }
      std::cout << rule << std::endl;
  }

  /**
   * Remove all trees from the forest.
   */
  void clear()
  {
      m_trees.clear();
  }

private:

  std::vector<DecisionTreeNode::ConstSharedPointer> m_trees;
};

/**
 * Writes a Forest to a binary file.
 */
inline void writeToFile( const Forest &forest, const std::string &filename )
{
    // Write the forest header to a newly opened file.
    std::ofstream out( filename, std::ios::binary | std::ios::out );
    out << 'f';

    // Serialize all trees.
    DecisionTreeWriter writer( out );
    for ( auto tree: forest )
    {
        tree->visit( writer );
    }

    // Close the file.
    out.close();
}


/**
 * Read a decision tree from a binary input stream.
 */
inline DecisionTreeNode::SharedPointer readTree( std::istream &in )
{
    // Read the node type ID.
    char t = 0;
    in >> t;

    // For internal nodes, parse the children and the node parameters.
    if ( t == 'i' )
    {
        // Read the node parameters.
        uint8_t splitFeature = 0;
        in >> splitFeature;
        double splitValue = 0;
        in.read( reinterpret_cast<char *>( &splitValue ), sizeof( splitValue ) );
        auto left  = readTree( in );
        auto right = readTree( in );
        return DecisionTreeNode::SharedPointer( new DecisionTreeInternalNode( splitFeature, splitValue, left, right ) );
    }

    // For leaf-nodes, parse the parameters.
    if ( t == 'l' )
    {
        // Read the label.
        char labelText;
        in >> labelText;
        if( labelText != 'T' && labelText != 'F' ) throw ParseError( "Unexpected leaf node label." );
        bool label = labelText == 'T';
        return DecisionTreeNode::SharedPointer( new DecisionTreeLeafNode( label ) );
    }

    throw ParseError( std::string( "Unexpected node label with value " ) + std::to_string( static_cast<unsigned int>( t ) ) + "." );
}

/**
 * Reads a Forest from a binary input stream. See also: loadForest().
 */
inline Forest readForest( std::istream &in )
{
    // Create an empty forest.
    Forest forest;

    // Read the header.
    assert( in.good() );
    if ( in.eof() ) throw ParseError( "Unexpected end of file." );
    char marker = 0;
    in >> marker;
    if ( marker != 'f' ) throw ParseError( "Unexpected header block." );

    // Parse trees until the end of the file is reached.
    while ( in.peek() != std::ifstream::traits_type::eof() )
    {
        forest.addTree( readTree( in ) );
    }

    return forest;
}

/**
 * Reads a Forest from a file.
 */
inline Forest loadForest( const std::string &name )
{
    // Open an input stream for the file.
    std::ifstream in( name, std::ios::binary | std::ios::in );
    if ( !in.good() ) throw ClientError( std::string( "Can't read file '" ) + name + "'." );

    // Read the forest.
    return readForest( in );
}

/**
 * An index for traversing points in a dataset in order of each feature.
 */
class FeatureIndex
{
public:

  typedef std::tuple< double, bool, DataPointID > Entry;
  typedef std::vector< Entry > SingleFeatureIndex;

  FeatureIndex( const TrainingDataSet &dataset ):
  m_trueCount( 0 )
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

          // Create entries for each point in the dataset, and count the 'true' labels.
          m_trueCount = 0;
          for ( DataPointID pointID( 0 ), end( dataset.size() ); pointID < end; ++pointID )
          {
              bool label = dataset.getLabel( pointID );
              if ( label ) ++m_trueCount;
              index.push_back( Entry( dataset.getFeatureValue( pointID, feature ), label,  pointID ) );
          }

          // Sort the index by feature value (the rest of the fields don't matter).
          std::sort( index.begin(), index.end() );
      }
  }

  SingleFeatureIndex::const_iterator featureBegin( unsigned int featureID ) const
  {
      return m_featureIndices[featureID].begin();
  }

  SingleFeatureIndex::const_iterator featureEnd( unsigned int featureID ) const
  {
      return m_featureIndices[featureID].end();
  }

  /**
   * Returns the number of features.
   */
  unsigned int getFeatureCount() const
  {
      return m_featureIndices.size();
  }

  /**
   * Returns the number of indexed points.
   */
  std::size_t size() const
  {
      return m_featureIndices[0].size();
  }

  /**
   * Returns the number of points labeled 'true'.
   */
  unsigned int getTrueCount() const
  {
      return m_trueCount;
  }

private:

  unsigned int m_trueCount;
  std::vector< SingleFeatureIndex >  m_featureIndices;
};


/**
 * Returns a random boolean, with probability of being true equal to an integer fraction.
 */
bool randomBool( unsigned int numerator, unsigned int denominator )
{
    assert( numerator <=  denominator );
    if ( numerator == denominator ) return true;

    static std::random_device dev;
    static std::mt19937 rng( dev() );
    std::uniform_int_distribution<std::mt19937::result_type> dist( 1, denominator );
    return dist(rng) <= numerator;
}

/**
 * A node in a decision tree, with special annotations for the training process.
 */
class TrainingTreeNode
{
public:

  typedef std::shared_ptr<TrainingTreeNode> SharedPointer;

  TrainingTreeNode( TrainingTreeNode *parent = nullptr ):
  m_parent             ( parent ),
  m_splitFeatureID     ( 0      ),
  m_splitValue         ( 0      ),
  m_totalCount         ( 0      ),
  m_trueCount          ( 0      ),
  m_totalCountLeftHalf ( 0      ),
  m_trueCountLeftHalf  ( 0      ),
  m_trueCountRightHalf ( 0      ),
  m_currentFeature     ( 0      ),
  m_bestSplitFeature   ( 0      ),
  m_bestSplitValue     ( 0      ),
  m_bestSplitGiniIndex ( 0      ),
  m_featuresToConsider ( 0      ),
  m_ignoringThisFeature( false  )
  {
  }

  /**
   * Allows this node to count a data point as one of its descendants, and returns the leaf-node that contains the point.
   * \return A pointer to the leaf node that contains the point (direct parent).
   */
  TrainingTreeNode *registerPoint( DataPointID pointID, const TrainingDataSet &dataset )
  {
      // Count the point if this is a leaf-node, and return this node as the parent.
      if ( !m_leftChild )
      {
          bool label = dataset.getLabel( pointID );
          assert( !m_rightChild );
          ++m_totalCount;
          if ( label ) ++m_trueCount;
          return this;
      }

      // Defer the registration to the correct child.
      // N.B. The comparison must be strictly-less. DO NOT change to <=, or the algorithm will break.
      if ( dataset.getFeatureValue( pointID, m_splitFeatureID ) < m_splitValue ) return m_leftChild->registerPoint( pointID, dataset );
      return m_rightChild->registerPoint( pointID, dataset );
  }

  // Debug
  void dump( unsigned int indent = 0 ) const
  {

      auto tab = std::string( indent, ' ' );
      std::cout << tab << "Node:" << std::endl
                << tab << "m_totalCount         = " << m_totalCount         << std::endl
                << tab << "m_trueCount          = " << m_trueCount          << std::endl
                << tab << "m_totalCountLeftHalf = " << m_totalCountLeftHalf << std::endl
                << tab << "m_trueCountLeftHalf  = " << m_trueCountLeftHalf  << std::endl
                << tab << "m_trueCountRightHalf = " << m_trueCountRightHalf << std::endl
                << tab << "m_currentFeature     = " << m_currentFeature     << std::endl
                << tab << "m_bestSplitFeature   = " << m_bestSplitFeature   << std::endl
                << tab << "m_bestSplitValue     = " << m_bestSplitValue     << std::endl
                << tab << "m_bestSplitGiniIndex = " << m_bestSplitGiniIndex << std::endl;

      if ( m_leftChild )
      {
          std::cout << tab << "Split feature #" << m_splitFeatureID << ", value = " <<  m_splitValue << std::endl;
          std::cout << tab << "Left: " << std::endl;
          m_leftChild->dump( indent + 1 );
          std::cout << tab << "Right: " << std::endl;
          m_rightChild->dump( indent + 1 );
      }
  }


  /**
   * Returns a stripped, non-training decision tree.
   */
  DecisionTreeNode::SharedPointer finalize()
  {
      // Build an internal node or a leaf node.
      if ( m_leftChild )
      {
          assert( m_rightChild );
          return DecisionTreeNode::SharedPointer( new DecisionTreeInternalNode( m_splitFeatureID, m_splitValue, m_leftChild->finalize(), m_rightChild->finalize() ) );
      }
      else
      {
          return DecisionTreeNode::SharedPointer( new DecisionTreeLeafNode( getLabel() ) );
      }
  }

  /**
   * Returns the most obvious label for this node.
   */
  bool getLabel() const
  {
      return m_totalCount < 2 * m_trueCount;
  }

  /**
   * Initializes the search for the optimal split.
   * \param featuresToConsider The number of randomly selected features that this node will consider during traversal.
   */
  void initializeOptimalSplitSearch( unsigned int featuresToConsider )
  {
      // Reset the point conts. Points will be re-counted during the point registration phase.
      m_trueCount          = 0;
      m_totalCount         = 0;

      // Reset the number of features that will be considered by this node.
      m_featuresToConsider = featuresToConsider;

      // Reset the best split found so far. This will be re-determined during the feature traversal phase.
      m_bestSplitFeature   = 0;
      m_bestSplitValue     = 0;
      m_bestSplitGiniIndex = std::numeric_limits<double>::max();

      // Initialize any children.
      if ( m_leftChild  ) m_leftChild ->initializeOptimalSplitSearch( featuresToConsider );
      if ( m_rightChild ) m_rightChild->initializeOptimalSplitSearch( featuresToConsider );
  }

  /**
   * Instructs this node and its children that a particular feature will be traversed in-order now.
   * \param featureID The ID of the feature that will be traversed.
   * \param featuresLeft The total number of features that still have to be traversed, including this one.
   */
  void startFeatureTraversal( unsigned int featureID, unsigned int featuresLeft )
  {
      // Start feature traversal in the children, if present.
      if ( m_leftChild )
      {
          assert( m_rightChild );
          m_leftChild->startFeatureTraversal ( featureID, featuresLeft );
          m_rightChild->startFeatureTraversal( featureID, featuresLeft );
      }
      else
      {
          // Reset the feature traversal statistics.
          m_lastVisitedValue    = std::numeric_limits<double>::min(); // Arbitrary.
          m_totalCountLeftHalf  = 0          ;
          m_trueCountLeftHalf   = 0          ;
          m_trueCountRightHalf  = m_trueCount;
          m_currentFeature      = featureID  ;

          // Determine whether or not this node will consider this feature during this pass.
          m_ignoringThisFeature = randomBool( m_featuresToConsider, featuresLeft );
          if ( m_ignoringThisFeature )
          {
              assert( m_featuresToConsider > 0 );
              --m_featuresToConsider; // Use up one 'credit'.
          }
      }
  }

  /**
   * Visit one point during the feature traversal phase.
   */
  void visitPoint( DataPointID, double featureValue, bool label )
  {
      // This must never be called on internal nodes.
      assert( !m_leftChild  );
      assert( !m_rightChild );

      // Do nothing if this node is not considering this feature.
      if ( m_ignoringThisFeature ) return;

      // If this is the start of a block of previously unseen feature values, calculate what the gain of a split would be.
      if ( ( featureValue != m_lastVisitedValue ) && ( m_totalCountLeftHalf > 0 ) )
      {
          // Compute the Gini gain, assuming a split is made at this point.
          auto totalCountRightHalf = m_totalCount - m_totalCountLeftHalf;
          auto giniLeft  = giniImpurity( m_trueCountLeftHalf , m_totalCountLeftHalf );
          auto giniRight = giniImpurity( m_trueCountRightHalf, totalCountRightHalf  );
          auto giniTotal = ( giniLeft * m_totalCountLeftHalf + giniRight * totalCountRightHalf ) / m_totalCount;

          // Save this split if it is the best one so far.
          if ( giniTotal < m_bestSplitGiniIndex )
          {
              m_bestSplitFeature   = m_currentFeature;
              m_bestSplitValue     = featureValue    ;
              m_bestSplitGiniIndex = giniTotal       ;
          }
      }

      // Count this point and its label. The point now belongs to the 'left half' of the pass for this feature.
      if ( label )
      {
          ++m_trueCountLeftHalf;
          --m_trueCountRightHalf;
      }
      ++m_totalCountLeftHalf;

      // Update the last visited value. This is necessary to detect the end of a block during the visit of the next point.
      m_lastVisitedValue = featureValue;
  }

  /**
   * Split the leaf nodes at the most optimal point, after all features have been traversed.
   * \return The number of splits made.
   */
  unsigned int split()
  {
      // If this is an interior node, split the children and quit.
      if ( m_leftChild )
      {
          assert( m_rightChild );
          auto l = m_leftChild ->split();
          auto r = m_rightChild->split();
          return l + r;
      }

      // Assert that this is a leaf node.
      assert( !m_leftChild  );
      assert( !m_rightChild );

      // Do not split if this node is completely pure (or empty).
      // TODO: the purity cutoff can be made more flexible.
      if ( m_trueCount == 0 || m_trueCount == m_totalCount ) return 0;

      // Split this node at the best point that was found.
      m_splitValue     = m_bestSplitValue;
      m_splitFeatureID = m_bestSplitFeature;
      m_leftChild .reset( new TrainingTreeNode( this ) );
      m_rightChild.reset( new TrainingTreeNode( this ) );

      return 1;
  }

  /**
   * Returns the path of this node from the root (debug purposes).
   */
  std::string getPath() const
  {
      if ( !m_parent ) return "root";
      return m_parent->getPath() + ( m_parent->isLeftChild( this ) ? ".L" : ".R" );
  }

private:

  bool isLeftChild( const TrainingTreeNode *node ) const
  {
      return m_leftChild.get() == node;
  }

  TrainingTreeNode *m_parent            ;
  SharedPointer     m_leftChild         ; // Null for leaf nodes.
  SharedPointer     m_rightChild        ; // Null for leaf nodes.
  unsigned int      m_splitFeatureID    ; // The ID of the feature at which this node is split. Only valid for internal nodes.
  double            m_splitValue        ; // The value at which this node is split, along the specified feature.
  unsigned int      m_totalCount        ; // Total number of points in this node.
  unsigned int      m_trueCount         ; // Total number of points labeled 'true' in this node.

  // Statistics used during traversal:
  double            m_lastVisitedValue   ;
  unsigned int      m_totalCountLeftHalf ; // Total number of points that have been visited during traversal of the current feature.
  unsigned int      m_trueCountLeftHalf  ; // Totol number of visited points labeled 'true'.
  unsigned int      m_trueCountRightHalf ; // Remaining unvisited points labeled 'true'.
  unsigned int      m_currentFeature     ; // The feature that is currently being traversed.
  unsigned int      m_bestSplitFeature   ; // Best feature for splitting found so far.
  double            m_bestSplitValue     ; // Best value to split at found so far.
  double            m_bestSplitGiniIndex ; // Gini-index of the best split point found so far (lowest index).
  unsigned int      m_featuresToConsider ; // The number of randomly chosen features that this node still has to consider during the feature traversal phase.
  bool              m_ignoringThisFeature; // Whether or not the currently traversed feature is taken into account by this node.

};

/**
 * Trains a single random decision tree at a time.
 */
class SingleTreeTrainer
{
public:

  SingleTreeTrainer( unsigned int maxDepth = std::numeric_limits<unsigned int>::max()  ):
  m_maxDepth( maxDepth )
  {
      // TODO:
      // Min samples split = 2
      // Min samples leaf = 1
      // Max features = sqrt(nfeatures)
      // Max leaf nodes = None
      // Min impurity decrease = 0.0
      // Bootstrap = True
  }

  DecisionTreeNode::SharedPointer train( const FeatureIndex &featureIndex, const TrainingDataSet &dataSet )
  {
      // Create an empty training tree.
      TrainingTreeNode root;

      // Create a list of pointers from data points to their current parent nodes.
      std::vector< TrainingTreeNode * > pointParents( featureIndex.size(), &root );

      // Split all leaf nodes in the tree until the depth limit is reached.
      for ( unsigned int depth = 0; depth < m_maxDepth; ++depth )
      {
          std::cout << "Depth = " << depth << std::endl;

          // Tell all nodes that a round of optimal split searching is starting.
          unsigned int featureCount = featureIndex.getFeatureCount();
          unsigned int featuresToConsider = std::ceil( std::sqrt( featureCount ) );
          assert( featuresToConsider > 0 );
          root.initializeOptimalSplitSearch( featuresToConsider );

          // Register all points with their respective parent nodes.
          for ( DataPointID pointID( 0 ), end( pointParents.size() ); pointID < end; ++pointID )
          {
              pointParents[pointID] = pointParents[pointID]->registerPoint( pointID, dataSet );
          }

          // Traverse all data points once for each feature, in order, so the tree nodes can find the best possible split for them.
          for ( unsigned int featureID = 0; featureID < featureCount; ++featureID ) // TODO: random trees should not use all features.
          {
              // Tell the tree that traversal is starting for this feature.
              root.startFeatureTraversal( featureID, featureCount - featureID );

              // Traverse all datapoints in order of this feature.
              for ( auto it( featureIndex.featureBegin( featureID ) ), end( featureIndex.featureEnd( featureID ) ); it != end; ++it )
              {
                  // Let the parent node of the data point know that it is being traversed.
                  auto &tuple = *it;
                  auto featureValue = std::get<0>( tuple );
                  auto label        = std::get<1>( tuple );
                  auto pointID      = std::get<2>( tuple );
                  pointParents[pointID]->visitPoint( pointID, featureValue, label );
              }
          }

          // Allow all leaf nodes to split, if necessary. Stop if there is no more improvement.
          auto splitCount = root.split();
          if ( splitCount == 0 ) break;
          std::cout << splitCount << " splits made." << std::endl;
      }

      // Return a stripped version of the training tree.
      return root.finalize();
  }

private:

  const unsigned int m_maxDepth;

};

#include "messagequeue.h"

/**
 * Trains a random binary forest classifier on a TrainingDataSet.
 */
class BinaryRandomForestTrainer
{

  // Used for distributing jobs to threads.
  class TrainingJob
  {
  public:

    TrainingJob( const TrainingDataSet &dataSet, const FeatureIndex &featureIndex, unsigned int maxDepth, bool stop ):
    dataSet     ( dataSet      ),
    featureIndex( featureIndex ),
    maxDepth    ( maxDepth     ),
    stop        ( stop         )
    {
    }

    const TrainingDataSet &dataSet     ;
    const FeatureIndex    &featureIndex;
    unsigned int           maxDepth    ;
    bool                   stop        ;
  };

public:

  /**
   * Constructor.
   * \param dataset A const reference to a training dataset. Modifying the set after construction of the trainer invalidates the trainer.
   * \param concurrentTrainers The maximum number of trees that may be trained concurrently.
   */
  BinaryRandomForestTrainer( unsigned maxDepth = std::numeric_limits<unsigned int>::max(), unsigned int treeCount = 10, unsigned int concurrentTrainers = 10 ):
  m_maxDepth( maxDepth ),
  m_trainerCount( concurrentTrainers ),
  m_treeCount( treeCount )
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
  Forest::SharedPointer train( TrainingDataSet::ConstSharedPointer dataset )
  {
      // Build the feature index that is common to all threads.
      FeatureIndex featureIndex( *dataset );

      // Create message queues for communicating with the worker threads.
      MessageQueue<TrainingJob                    > jobOutbox;
      MessageQueue<DecisionTreeNode::SharedPointer> treeInbox;

      // Start the worker threads.
      std::vector<std::thread> workers;
      for ( unsigned int i = 0; i < m_trainerCount; ++i )
      {
          workers.push_back( std::thread( &BinaryRandomForestTrainer::workerThread, i, &jobOutbox, &treeInbox ) );
      }

      // Create jobs for all trees.
      for ( unsigned int i = 0; i < m_treeCount; ++i )
          jobOutbox.send( TrainingJob( *dataset, featureIndex, m_maxDepth, false ) );

      // Create 'stop' messages for all threads, to be picked up after all the work is done.
      for ( unsigned int i = 0; i < workers.size(); ++i )
           jobOutbox.send( TrainingJob( *dataset, featureIndex, 0, true ) );

      // Wait for all the trees to come in, and add them to the forest.
      Forest::SharedPointer forest( new Forest );
      for ( unsigned int i = 0; i < m_treeCount; ++i )
          forest->addTree( treeInbox.receive() );

      // Wait for all the threads to join.
      for ( auto &worker: workers ) worker.join();

      // Return the trained model.
      return forest;
  }

private:

  static void workerThread( unsigned int workerID, MessageQueue<TrainingJob> *jobInbox, MessageQueue<DecisionTreeNode::SharedPointer> *treeOutbox )
  {
      // Train trees until it is time to stop.
      unsigned int jobsPickedUp = 0;
      while ( true )
      {
          // Get an assignment or stop message from the queue.
          TrainingJob job = jobInbox->receive();
          if ( job.stop ) break;
          ++jobsPickedUp;
          std::cout << "Worker #" << workerID << ": job " << jobsPickedUp << " picked up." << std::endl;

          // Train a tree and send it to the main thread.
          SingleTreeTrainer trainer( job.maxDepth );
          treeOutbox->send( trainer.train( job.featureIndex, job.dataSet ) );
          std::cout << "Worker #" << workerID << ": job " << jobsPickedUp << " finished." << std::endl;
      }

      std::cout << "Worker #" << workerID << " finished." << std::endl;
  }

  unsigned int m_maxDepth    ;
  unsigned int m_trainerCount;
  unsigned int m_treeCount   ;
};

#endif // DATAMODEL_H
