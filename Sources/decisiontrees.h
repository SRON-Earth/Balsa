#ifndef DECISIONTREES_H
#define DECISIONTREES_H

#include <cassert>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <iomanip>

#include "exceptions.h"
#include "datarepresentation.h"

struct DecisionTreeNode
{
    // DecisionTreeNode() : leftChildID(0), rightChildID(0), splitFeatureID(0), splitValue(0.0), label(false) {}
    unsigned int leftChildID   ;
    unsigned int rightChildID  ;
    unsigned int splitFeatureID;
    double       splitValue    ;
    bool         label         ;
};

inline bool isLeafNode( const DecisionTreeNode & node )
{
    return !node.leftChildID && !node.rightChildID;
}

template <typename T>
T read( std::istream & stream )
{
    T result;
    stream.read( reinterpret_cast<char *>( &result ), sizeof( T ) );
    return result;
}

inline DecisionTreeNode readDecisionTreeNode( std::istream & in )
{
    DecisionTreeNode node;
    node.leftChildID    = read<std::uint32_t>( in );
    node.rightChildID   = read<std::uint32_t>( in );
    node.splitFeatureID = read<std::uint32_t>( in );
    node.splitValue     = read<double       >( in );
    node.label          = read<bool         >( in );
    return node;
}

template <typename T>
void write( std::ostream & os, const T & value )
{
    os.write( reinterpret_cast<const char *>( &value ), sizeof( T ) );
}

inline void writeDecisionTreeNode( std::ostream & os, const DecisionTreeNode & node )
{
    write<std::uint32_t>( os, node.leftChildID    );
    write<std::uint32_t>( os, node.rightChildID   );
    write<std::uint32_t>( os, node.splitFeatureID );
    write<double       >( os, node.splitValue     );
    write<bool         >( os, node.label          );
}

class DecisionTree
{
public:

  typedef std::shared_ptr <      DecisionTree> SharedPointer     ;
  typedef std::shared_ptr <const DecisionTree> ConstSharedPointer;
  typedef std::vector<DecisionTreeNode>::const_iterator ConstIterator;

  DecisionTree( std::size_t reservation = 0 )
  {
      m_nodes.reserve( reservation );
  }

  ConstIterator begin() const
  {
      return m_nodes.begin();
  }

  ConstIterator end() const
  {
      return m_nodes.end();
  }

  /**
   * Returns the number of nodes in this tree.
   */
  unsigned int getNodeCount() const
  {
      return m_nodes.size();
  }

  /**
   * Returns the depth of this tree.
   */
  unsigned int getDepth() const
  {
      return getDepth( 0 );
  }

  inline DecisionTreeNode & getNode( unsigned int nodeID )
  {
      return m_nodes.at( nodeID );
  }

  inline const DecisionTreeNode & getNode( unsigned int nodeID ) const
  {
      return m_nodes.at( nodeID );
  }

  bool classify( const DataSet &dataSet, DataPointID pointID ) const
  {
      return classify( dataSet, pointID, 0 );
  }

  unsigned int addNode( const DecisionTreeNode & node )
  {
      m_nodes.push_back( node );
      return m_nodes.size() - 1;
  }

  void dump( unsigned int indent = 0 ) const
  {
      dump( indent, 0 );
  }

private:

  void dump( unsigned int indent, unsigned int nodeID ) const
  {
      auto tab = std::string( indent, ' ' );

      const DecisionTreeNode & node = getNode( nodeID );
      std::cout << tab << "Node #" << nodeID << ", feature #" << node.splitFeatureID
                << ", split value = " << std::setprecision( 17 )
                << node.splitValue << ", label = "
                << ( node.label ? "TRUE" : "FALSE" ) << std::endl;
      if (node.leftChildID || node.rightChildID)
      {
          // Internal node.
          std::cout << tab << "Left:" << std::endl;
          dump( indent + 1, node.leftChildID );
          std::cout << tab << "Right:" << std::endl;
          dump( indent + 1, node.rightChildID );
      }
      // if (node.leftChildID || node.rightChildID)
      // {
      //     // Internal node.
      //     std::cout << tab << "Feature #" << node.splitFeatureID
      //               << ", split value = " << std::setprecision( 17 )
      //               << node.splitValue << std::endl;
      //     std::cout << tab << "Left:" << std::endl;
      //     dump( indent + 1, node.leftChildID );
      //     std::cout << tab << "Right:" << std::endl;
      //     dump( indent + 1, node.rightChildID );
      // }
      // else
      // {
      //     // Leaf node.
      //     std::cout << tab << ( node.label ? "TRUE" : "FALSE" ) << std::endl;
      // }
  }

  bool classify( const DataSet &dataSet, DataPointID pointID, unsigned int nodeID ) const
  {
      const DecisionTreeNode & node = getNode( nodeID );

      if (node.leftChildID || node.rightChildID)
      {
          // Internal node
          if ( dataSet.getFeatureValue( pointID, node.splitFeatureID ) < node.splitValue )
              return classify( dataSet, pointID, node.leftChildID );
          return classify( dataSet, pointID, node.rightChildID );
      }

      return node.label;
  }

  unsigned int getDepth( unsigned int nodeID ) const
  {
      const DecisionTreeNode & node = getNode( nodeID );
      const unsigned int depthLeft  = ( node.leftChildID  == 0 ) ? 0 : getDepth( node.leftChildID  );
      const unsigned int depthRight = ( node.rightChildID == 0 ) ? 0 : getDepth( node.rightChildID );
      return 1 + std::max( depthLeft, depthRight );
  }

  std::vector<DecisionTreeNode> m_nodes;
};

inline DecisionTree::ConstSharedPointer readDecisionTree( std::istream & in )
{
    // Create an empty decision tree.
    DecisionTree::SharedPointer tree( new DecisionTree() );

    // Read the header.
    assert( in.good() );
    if ( in.eof() ) throw ParseError( "Unexpected end of file." );
    char marker = read<char>( in );
    if ( marker != 't' ) throw ParseError( "Unexpected header block." );

    // Read the node count.
    if ( in.eof() ) throw ParseError( "Unexpected end of file." );
    std::size_t nodeCount = read<std::uint64_t>( in );

    // Parse tree nodes.
    for ( ; nodeCount > 0; --nodeCount )
    {
        tree->addNode( readDecisionTreeNode( in ) );
    }

    return tree;
}

inline void writeDecisionTree( std::ostream & os, const DecisionTree & tree )
{
    write<char>( os, 't' );
    write<std::uint64_t>( os, tree.getNodeCount() );
    for ( auto const & node : tree )
    {
        writeDecisionTreeNode( os, node );
    }
}

/**
 * Writes a decision tree to a binary file.
 */
inline void writeToFile( const DecisionTree & tree, const std::string & filename )
{
    // Serialize the tree.
    std::ofstream out( filename, std::ios::binary | std::ios::out );
    writeDecisionTree( out, tree );
};

/**
 * An ensemble of (usually randomized) decision trees.
 */
class Forest
{
public:

  typedef std::shared_ptr<Forest> SharedPointer;
  typedef std::vector<DecisionTree::ConstSharedPointer>::const_iterator ConstIterator;

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

  // /**
  //  * Return the classification of a data point.
  //  */
  // bool classify( const DataPoint &point ) const
  // {
  //     unsigned int trueCount = 0;
  //     // for ( auto tree: m_trees ) if ( tree->classify( point ) ) ++trueCount;
  //     return trueCount >= m_trees.size() / 2; // TODO: weighted voting based on quality of out-of-bag predictions of subtrees?
  // }

  /**
   * Add a tree to the forest.
   * \param tree The root node of the tree.
   */
  void addTree( DecisionTree::ConstSharedPointer tree )
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
      for ( auto const & tree: m_trees )
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

  std::vector<DecisionTree::ConstSharedPointer> m_trees;
};

/**
 * Reads a Forest from a binary input stream. See also: loadForest().
 */
inline Forest::SharedPointer readForest( std::istream &in )
{
    // Create an empty forest.
    Forest::SharedPointer forest( new Forest() );

    // Read the header.
    assert( in.good() );
    if ( in.eof() ) throw ParseError( "Unexpected end of file." );
    char marker = 0;
    in >> marker;
    if ( marker != 'f' ) throw ParseError( "Unexpected header block." );

    // Parse trees until the end of the file is reached.
    while ( in.peek() != std::ifstream::traits_type::eof() )
    {
        forest->addTree( readDecisionTree( in ) );
        assert( in.good() );
    }

    return forest;
}

/**
 * Writes a Forest to a binary output stream. See also: storeForest().
 */
inline void writeForest( std::ostream & out, const Forest &forest )
{
    // Write the forest header.
    out << 'f';

    // Serialize all trees.
    for ( auto const & tree : forest )
    {
        writeDecisionTree( out, *tree );
    }
}

/**
 * Reads a Forest from a file.
 */
inline Forest::SharedPointer loadForest( const std::string &filename )
{
    // Open an input stream for the file.
    std::ifstream in( filename, std::ios::binary | std::ios::in );
    if ( !in.good() ) throw ClientError( std::string( "Can't read file '" ) + filename + "'." );

    // Read the forest.
    return readForest( in );
}

/**
 * Writes a Forest to a file.
 */
inline void storeForest( const Forest &forest, const std::string &filename )
{
    // Open an output stream for the file.
    std::ofstream out( filename, std::ios::binary | std::ios::out );
    if ( !out.good() ) throw SupplierError( std::string( "Can't create file '" ) + filename + "'." );

    // Write the forest.
    writeForest( out, forest );
}

#endif // DECISIONTREES_H
