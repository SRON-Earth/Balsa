#ifndef DECISIONTREES_H
#define DECISIONTREES_H

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

#include "classifier.h"
#include "datarepresentation.h"
#include "exceptions.h"

/**
 * A decision tree.
 */
template <typename FeatureIterator, typename OutputIterator>
class DecisionTree: public Classifier<FeatureIterator, OutputIterator>
{
public:
  typedef std::shared_ptr<DecisionTree<FeatureIterator, OutputIterator>> SharedPointer;
  typedef std::shared_ptr<const DecisionTree<FeatureIterator, OutputIterator>> ConstSharedPointer;

  typedef typename std::iterator_traits<FeatureIterator>::value_type FeatureValueType;
  typedef typename std::iterator_traits<OutputIterator>::value_type LabelValueType;

  using typename Classifier<FeatureIterator, OutputIterator>::VoteTable;

  typedef unsigned int NodeID;

  struct DecisionTreeNode
  {
      unsigned int      leftChildID   ;
      unsigned int      rightChildID  ;
      unsigned char     splitFeatureID;
      FeatureValueType  splitValue    ;
      LabelValueType    label         ;
  };

  typedef typename std::vector<DecisionTreeNode>::const_iterator ConstIterator;

  /**
   * Construct an empty decision tree.
   */
  explicit DecisionTree( std::size_t featureCount )
  : Classifier<FeatureIterator, OutputIterator>( featureCount )
  {
  }

  /**
   * Construct a decision tree from a sequence of decision tree nodes.
   */
  template <typename T>
  DecisionTree( std::size_t featureCount, T first, T last )
  : Classifier<FeatureIterator, OutputIterator>( featureCount )
  , m_nodes( first, last )
  {
  }

  /**
   * Return an iterator to the beginning of the collection of nodes in this
   * tree.
   */
  ConstIterator begin() const
  {
      return m_nodes.begin();
  }

  /**
   * Return an iterator to the end of the collection of nodes in this tree.
   */
  ConstIterator end() const
  {
      return m_nodes.end();
  }

  /**
   * Pre-allocate space for the given number of nodes.
   */
  void reserve( std::size_t size )
  {
      m_nodes.reserve( size );
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
      return getDepth( NodeID( 0 ) );
  }

  /**
   * Return a reference to the node with the specified ID.
   */
  DecisionTreeNode & operator[]( unsigned int nodeID )
  {
      return m_nodes[nodeID];
  }

  /**
   * Return a const reference to the node with the specified ID.
   */
  const DecisionTreeNode & operator[]( unsigned int nodeID ) const
  {
      return m_nodes[nodeID];
  }

  /**
   * Bulk-classifies a sequence of data points.
   */
  void classify( FeatureIterator pointsStart, FeatureIterator pointsEnd,
      OutputIterator labels ) const;

  /**
   * Add a new node to the tree and return its ID.
   */
  unsigned int addNode( const DecisionTreeNode & node )
  {
      m_nodes.push_back( node );
      return m_nodes.size() - 1;
  }

  /**
   * Print the tree for debugging purposes.
   */
  void dump( unsigned int indent = 0 ) const
  {
      dump( NodeID( 0 ), indent );
  }

  /**
  * Bulk-classifies a set of points, adding a vote (+1) to the vote table for
  * each point of which the label is 'true'.
  */
  unsigned int classifyAndVote( FeatureIterator pointsStart,
      FeatureIterator pointsEnd, VoteTable & table ) const;

private:

  void recursiveClassifyVote( std::vector<DataPointID>::iterator pointIDsStart,
    std::vector<DataPointID>::iterator pointIDsEnd, FeatureIterator pointsStart,
    VoteTable & result, NodeID currentNodeID ) const;
  void dump( NodeID nodeID, unsigned int indent ) const;
  unsigned int getDepth( NodeID nodeID ) const;

  std::vector<DecisionTreeNode> m_nodes;
};

template <typename FeatureIterator, typename OutputIterator>
void DecisionTree<FeatureIterator, OutputIterator>::dump( NodeID nodeID, unsigned int indent ) const
{
    auto tab = std::string( indent, ' ' );

    const DecisionTreeNode & node = m_nodes[nodeID];
    if (node.leftChildID || node.rightChildID)
    {
        // Internal node.
        std::cout << tab << "Node #" << nodeID
                  << " Feature #" << static_cast<unsigned int>( node.splitFeatureID )
                  << ", split value = " << std::setprecision( 17 )
                  << node.splitValue << std::endl;
        std::cout << tab << "Left:" << std::endl;
        dump( node.leftChildID, indent + 1 );
        std::cout << tab << "Right:" << std::endl;
        dump( node.rightChildID, indent + 1 );
    }
    else
    {
        // Leaf node.
        std::cout << tab << "Node #" << nodeID << " " << ( node.label ? "TRUE" : "FALSE" ) << std::endl;
    }
}

template <typename FeatureIterator, typename OutputIterator>
void DecisionTree<FeatureIterator, OutputIterator>::classify( FeatureIterator pointsStart, FeatureIterator pointsEnd,
    OutputIterator labels ) const
{
    // Check the dimensions of the input data.
    auto rawFeatureCount = std::distance( pointsStart, pointsEnd );
    auto featureCount    = this->getFeatureCount();
    assert( rawFeatureCount > 0 );
    assert( ( rawFeatureCount % featureCount ) == 0 );

    // Create a table for the label votes.
    unsigned int pointCount = rawFeatureCount / featureCount;
    VoteTable voteCounts( typename VoteTable::value_type( 0 ), pointCount );

    // Bulk-classify all points.
    classifyAndVote( pointsStart, pointsEnd, voteCounts );

    // Generate the labels.
    for ( auto voteCount : voteCounts ) *labels++ = voteCount > 0;
}

template <typename FeatureIterator, typename OutputIterator>
unsigned int DecisionTree<FeatureIterator, OutputIterator>::classifyAndVote( FeatureIterator pointsStart, FeatureIterator pointsEnd,
    VoteTable & table ) const
{
    // Check the dimensions of the input data.
    auto rawFeatureCount = std::distance( pointsStart, pointsEnd );
    auto featureCount    = this->getFeatureCount();
    assert( rawFeatureCount > 0 );
    assert( ( rawFeatureCount % featureCount ) == 0 );

    // Determine the number of points in the input data.
    unsigned int pointCount = rawFeatureCount / featureCount;

    // Create a list containing all datapoint IDs (0, 1, 2, etc.).
    std::vector<DataPointID> pointIDs( pointCount );
    std::iota( pointIDs.begin(), pointIDs.end(), 0 );

    // Recursively partition the list of point IDs according to the interior
    // node criteria, and classify them by the leaf node labels.
    recursiveClassifyVote( pointIDs.begin(), pointIDs.end(), pointsStart, table, NodeID( 0 ) );

    return 1;
}

template <typename FeatureIterator, typename OutputIterator>
void DecisionTree<FeatureIterator, OutputIterator>::recursiveClassifyVote( std::vector<DataPointID>::iterator pointIDsStart,
    std::vector<DataPointID>::iterator pointIDsEnd, FeatureIterator pointsStart, VoteTable & voteTable,
    NodeID currentNodeID ) const
{
    // If the current node is an interior node, split the points along the split value, and classify both halves.
    auto &currentNode = m_nodes[currentNodeID];
    if ( currentNode.leftChildID > 0 )
    {
        // Extract the split limit and split dimension of this node.
        auto splitValue = currentNode.splitValue;
        auto featureID  = currentNode.splitFeatureID;

        // Retrieve feature count.
        auto featureCount = this->getFeatureCount();

        // Split the point IDs in two halves: points that lie below and points that lie on or above the feature split value.
        auto pointIsBelowLimit = [&pointsStart,featureCount,splitValue,featureID]( const unsigned int &pointID )
        {
            return pointsStart[featureCount * pointID + featureID] < splitValue;
        };
        auto secondHalf = std::partition( pointIDsStart, pointIDsEnd, pointIsBelowLimit );

        // Recursively classify both halves.
        recursiveClassifyVote( pointIDsStart, secondHalf, pointsStart, voteTable, currentNode.leftChildID  );
        recursiveClassifyVote( secondHalf , pointIDsEnd , pointsStart, voteTable, currentNode.rightChildID );
    }

    // If the current node is a leaf node (and the label is 'true'), increment the true count in the output.
    else
    {
        if ( currentNode.label )
        {
            for ( auto it( pointIDsStart ), end( pointIDsEnd ); it != end; ++it )
            {
                assert( *it < voteTable.size() );
                ++voteTable[*it];
            }
        }
    }
}

template <typename FeatureIterator, typename OutputIterator>
unsigned int DecisionTree<FeatureIterator, OutputIterator>::getDepth( NodeID nodeID ) const
{
    const DecisionTreeNode & node = m_nodes[nodeID];
    const unsigned int depthLeft  = ( node.leftChildID  == 0 ) ? 0 : getDepth( node.leftChildID  );
    const unsigned int depthRight = ( node.rightChildID == 0 ) ? 0 : getDepth( node.rightChildID );
    return 1 + std::max( depthLeft, depthRight );
}

/**
 * Determine whether a decision tree node is a leaf node.
 */
template <typename FeatureIterator, typename OutputIterator>
inline bool isLeafNode( const typename DecisionTree<FeatureIterator, OutputIterator>::DecisionTreeNode & node )
{
    return node.leftChildID == 0;
}

/**
 * Deserialize a POD (Plain Old Data) value from a binary input stream.
 */
template <typename T>
inline T readPODValue( std::istream & stream )
{
    T result;
    stream.read( reinterpret_cast<char *>( &result ), sizeof( T ) );
    return result;
}

/**
 * Serialize a POD (Plain Old Data) value to a binary input stream.
 */
template <typename T>
void writePODValue( std::ostream & os, const T & value )
{
    os.write( reinterpret_cast<const char *>( &value ), sizeof( T ) );
}

/**
 * Read a decision tree from a binary input stream.
 */
template <typename FeatureIterator, typename OutputIterator>
typename DecisionTree<FeatureIterator, OutputIterator>::SharedPointer readDecisionTree( std::istream & in )
{
    // Read the header.
    assert( in.good() );
    if ( in.eof() ) throw ParseError( "Unexpected end of file." );
    char marker = readPODValue<char>( in );
    if ( in.eof() ) throw ParseError( "Unexpected end of file." );
    if ( marker != 't' ) throw ParseError( "Unexpected header block." );

    // Read the feature count.
    if ( in.eof() ) throw ParseError( "Unexpected end of file." );
    std::size_t featureCount = readPODValue<std::uint8_t>( in );

    // Create an empty decision tree.
    typedef DecisionTree<FeatureIterator, OutputIterator> DecisionTreeType;
    typename DecisionTreeType::SharedPointer tree( new DecisionTreeType( featureCount ) );

    // Read the node count.
    if ( in.eof() ) throw ParseError( "Unexpected end of file." );
    std::size_t nodeCount = readPODValue<std::uint64_t>( in );

    // Calculate the size of one tree node.
    typedef typename DecisionTreeType::DecisionTreeNode DecisionTreeNodeType;
    std::size_t rawNodeSize  = sizeof( DecisionTreeNodeType().leftChildID    )
                             + sizeof( DecisionTreeNodeType().rightChildID   )
                             + sizeof( DecisionTreeNodeType().splitFeatureID )
                             + sizeof( DecisionTreeNodeType().splitValue     )
                             + sizeof( DecisionTreeNodeType().label          );

    // Allocate a buffer and read the raw tree data into it.
    auto rawTreeSize = rawNodeSize * nodeCount;
    std::unique_ptr<char[]> buffer(new char[rawTreeSize]);
    in.read( &(buffer[0]), rawTreeSize );

    if ( in.fail() ) throw ParseError( "Could not read data." );

    // Parse tree nodes.
    char *rawNode = &(buffer[0]);
    for ( std::size_t nodeID = 0; nodeID < nodeCount; ++nodeID )
    {
        DecisionTreeNodeType node;

        node.leftChildID = *reinterpret_cast<decltype( DecisionTreeNodeType().leftChildID ) *>( rawNode );
        rawNode += sizeof( DecisionTreeNodeType().leftChildID );

        node.rightChildID = *reinterpret_cast<decltype( DecisionTreeNodeType().rightChildID ) *>( rawNode );
        rawNode += sizeof( DecisionTreeNodeType().rightChildID );

        node.splitFeatureID = *reinterpret_cast<decltype( DecisionTreeNodeType().splitFeatureID ) *>( rawNode );
        rawNode += sizeof( DecisionTreeNodeType().splitFeatureID );

        node.splitValue = *reinterpret_cast<decltype( DecisionTreeNodeType().splitValue ) *>( rawNode );
        rawNode += sizeof( DecisionTreeNodeType().splitValue );

        node.label = *reinterpret_cast<decltype( DecisionTreeNodeType().label ) *>( rawNode );
        rawNode += sizeof( DecisionTreeNodeType().label );

        tree->addNode( node );
    }

    return tree;
}

template <typename LabelType>
void writeLabel( std::ostream & os, LabelType label );

template <>
void writeLabel( std::ostream & os, bool label )
{
    writePODValue<std::uint8_t>( os, label );
}

template <>
void writeLabel( std::ostream & os, std::uint8_t label )
{
    writePODValue<std::uint8_t>( os, label );
}

template <typename FeatureType>
void writeSplitValue( std::ostream & os, FeatureType splitValue );

template <>
void writeSplitValue( std::ostream & os, float splitValue )
{
    writePODValue<float>( os, splitValue );
}

template <>
void writeSplitValue( std::ostream & os, double splitValue )
{
    writePODValue<double>( os, splitValue );
}

/**
 * Write a decision tree to a binary output stream.
 */
template <typename FeatureIterator, typename OutputIterator>
void writeDecisionTree( std::ostream & os, const DecisionTree<FeatureIterator, OutputIterator> & tree )
{
    writePODValue<char>( os, 't' );
    writePODValue<std::uint8_t>( os, tree.getFeatureCount() );
    writePODValue<std::uint64_t>( os, tree.getNodeCount() );
    for ( auto const & node : tree )
    {
        writePODValue<std::uint32_t>( os, node.leftChildID    );
        writePODValue<std::uint32_t>( os, node.rightChildID   );
        writePODValue<std::uint8_t >( os, node.splitFeatureID );
        writeSplitValue( os, node.splitValue );
        writeLabel( os, node.label );
    }
}

/**
 * Read a decision tree from a binary file.
 */
template <typename FeatureIterator, typename OutputIterator>
typename DecisionTree<FeatureIterator, OutputIterator>::SharedPointer loadDecisionTree( const std::string & filename )
{
    // Serialize the tree.
    std::ifstream in( filename, std::ios::binary );
    return readDecisionTree<FeatureIterator, OutputIterator>( in );
};

/**
 * Writes a decision tree to a binary file.
 */
template <typename FeatureIterator, typename OutputIterator>
void storeDecisionTree( const DecisionTree<FeatureIterator, OutputIterator> & tree, const std::string & filename )
{
    // Serialize the tree.
    std::ofstream out( filename, std::ios::binary | std::ios::out );
    writeDecisionTree( out, tree );
};

#endif // DECISIONTREES_H
