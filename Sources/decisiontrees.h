#ifndef DECISIONTREES_H
#define DECISIONTREES_H

#include <iostream>
#include <fstream>
#include <iomanip>

#include "exceptions.h"
#include "datarepresentation.h"

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
   * Returns the number of nodes in this tree.
   */
  virtual unsigned int getNodeCount() const = 0;

  /**
   * Returns the depth of this tree.
   */
  virtual unsigned int getDepth() const = 0;

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

  /**
   * Returns the number of nodes in this tree.
   */
  unsigned int getNodeCount() const
  {
      return 1 + m_leftChild->getNodeCount() + m_rightChild->getNodeCount();
  }

  /**
   * Returns the depth of this tree.
   */
  unsigned int getDepth() const
  {
      return 1 + std::max( m_leftChild->getDepth(), m_rightChild->getDepth() );
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

  /**
   * Returns the label of (all points in) this node.
   */
  bool getLabel() const
  {
      return m_label;
  }

  /**
   * Returns the number of nodes in this tree.
   */
  unsigned int getNodeCount() const
  {
      return 1;
  }

  /**
   * Returns the depth of this tree.
   */
  unsigned int getDepth() const
  {
      return 1;
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
   * Returns the number of nodes in the largest tree.
   */
  unsigned int getMaximumNodeCount() const
  {
      unsigned int maximum = 0;
      for ( auto &tree: m_trees ) maximum = std::max( maximum, tree->getNodeCount() );
      return maximum;
  }

  /**
   * Returns the depth of the largest tree.
   */
  unsigned int getMaximumDepth() const
  {
      unsigned int maximum = 0;
      for ( auto &tree: m_trees ) maximum = std::max( maximum, tree->getDepth() );
      return maximum;
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

#endif // DECISIONTREES_H
