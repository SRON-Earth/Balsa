#include <cassert>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <numeric>

#include "decisiontrees.h"
#include "exceptions.h"

namespace
{
  template <typename T>
  T read( std::istream & stream )
  {
      T result;
      stream.read( reinterpret_cast<char *>( &result ), sizeof( T ) );
      return result;
  }

  template <typename T>
  void write( std::ostream & os, const T & value )
  {
      os.write( reinterpret_cast<const char *>( &value ), sizeof( T ) );
  }

  DecisionTree::DecisionTreeNode readDecisionTreeNode( std::istream & in )
  {
      DecisionTree::DecisionTreeNode node;
      node.leftChildID    = read<std::uint32_t>( in );
      node.rightChildID   = read<std::uint32_t>( in );
      node.splitFeatureID = read<std::uint8_t >( in );
      node.splitValue     = read<double       >( in );
      node.label          = read<bool         >( in );
      return node;
  }

  void writeDecisionTreeNode( std::ostream & os, const DecisionTree::DecisionTreeNode & node )
  {
      write<std::uint32_t>( os, node.leftChildID    );
      write<std::uint32_t>( os, node.rightChildID   );
      write<std::uint8_t >( os, node.splitFeatureID );
      write<double       >( os, node.splitValue     );
      write<bool         >( os, node.label          );
  }
} // Anonymous namespace.

void DecisionTree::dump( NodeID nodeID, unsigned int indent ) const
{
    auto tab = std::string( indent, ' ' );

    const DecisionTreeNode & node = m_nodes[nodeID];
    if (node.leftChildID || node.rightChildID)
    {
        // Internal node.
        std::cout << tab << "Node #" << nodeID << " Feature #" << static_cast<unsigned int>( node.splitFeatureID )
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


bool DecisionTree::classify( NodeID nodeID, const DataSet &dataSet, DataPointID pointID ) const
{
    const DecisionTreeNode & node = m_nodes[nodeID];

    //  Recurse if this is an internal node.
    if (node.leftChildID)
    {
        if ( dataSet.getFeatureValue( pointID, node.splitFeatureID ) < node.splitValue )
            return classify( node.leftChildID, dataSet, pointID );
        return classify( node.rightChildID, dataSet, pointID );
    }

    // Return the label if it is a leaf node.
    return node.label;
}

void DecisionTree::recursiveClassifyVote( std::vector<unsigned int>::iterator pointsBegin, std::vector<unsigned int>::iterator pointsEnd, const DataSet &dataSet, std::vector<unsigned int> &voteTable, unsigned int currentNodeID ) const
{
    // If the current node is an interior node, split the points along the split value, and classify both halves.
    auto &currentNode = m_nodes[currentNodeID];
    if ( currentNode.leftChildID > 0 )
    {
        // Extract the split limit and split dimension of this node.
        auto splitValue = currentNode.splitValue;
        auto featureID  = currentNode.splitFeatureID;

        // Split the point IDs in two halves: points that lie below and points that lie on or above the feature split value.
        auto pointIsBelowLimit = [&dataSet,splitValue,featureID]( const unsigned int &pointID )
        {
            return dataSet.getFeatureValue( pointID, featureID ) < splitValue;
        };
        auto secondHalf = std::partition( pointsBegin, pointsEnd, pointIsBelowLimit );

        // Recursively classify both halves.
        recursiveClassifyVote( pointsBegin, secondHalf, dataSet, voteTable, currentNode.leftChildID  );
        recursiveClassifyVote( secondHalf , pointsEnd , dataSet, voteTable, currentNode.rightChildID );
    }

    // If the current node is a leaf node (and the label is 'true'), increment the true count in the output.
    else
    {
        if ( currentNode.label )
        {
            for ( auto it( pointsBegin ), end( pointsEnd ); it != end; ++it )
            {
                ++voteTable[*it];
            }
        }
    }
}

void DecisionTree::classifyVote( const DataSet &dataSet, std::vector<unsigned int> &result ) const
{
    // Create a list containing all datapoint IDs (0, 1, 2, etc.).
    std::vector<unsigned int> pointIDs( dataSet.size() );
    std::iota( pointIDs.begin(), pointIDs.end(), 0 );

    // Recursively partition the list of point IDs according to the interior node criteria, and classify them by the leaf node labels.
    recursiveClassifyVote( pointIDs.begin(), pointIDs.end(), dataSet, result, 0 );
}

unsigned int DecisionTree::getDepth( NodeID nodeID ) const
{
    const DecisionTreeNode & node = m_nodes[nodeID];
    const unsigned int depthLeft  = ( node.leftChildID  == 0 ) ? 0 : getDepth( node.leftChildID  );
    const unsigned int depthRight = ( node.rightChildID == 0 ) ? 0 : getDepth( node.rightChildID );
    return 1 + std::max( depthLeft, depthRight );
}

DecisionTree::SharedPointer readDecisionTree( std::istream & in )
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

    // Calculate the size of one tree node.
    std::size_t rawNodeSize  = sizeof( DecisionTree::DecisionTreeNode().leftChildID    )
                             + sizeof( DecisionTree::DecisionTreeNode().rightChildID   )
                             + sizeof( DecisionTree::DecisionTreeNode().splitFeatureID )
                             + sizeof( DecisionTree::DecisionTreeNode().splitValue     )
                             + sizeof( DecisionTree::DecisionTreeNode().label          );

    // Allocate a buffer and read the raw tree data into it.
    auto rawTreeSize = rawNodeSize * nodeCount;
    char buffer[rawTreeSize];
    in.read( buffer, rawTreeSize );

    if ( in.fail() ) throw ParseError( "Could not read data." );

    // Parse tree nodes.
    char *rawNode = buffer;
    for ( std::size_t nodeID = 0; nodeID < nodeCount; ++nodeID )
    {
        DecisionTree::DecisionTreeNode node;
        node.leftChildID    = *reinterpret_cast<std::uint32_t *>( rawNode ); rawNode += sizeof( std::uint32_t );
        node.rightChildID   = *reinterpret_cast<std::uint32_t *>( rawNode ); rawNode += sizeof( std::uint32_t );
        node.splitFeatureID = *reinterpret_cast<std::uint8_t  *>( rawNode ); rawNode += sizeof( std::uint8_t  );
        node.splitValue     = *reinterpret_cast<double        *>( rawNode ); rawNode += sizeof( double        );
        node.label          = *reinterpret_cast<bool          *>( rawNode ); rawNode += sizeof( bool          );
        tree->addNode( node );
    }

    return tree;
}

void writeDecisionTree( std::ostream & os, const DecisionTree & tree )
{
    write<char>( os, 't' );
    write<std::uint64_t>( os, tree.getNodeCount() );
    for ( auto const & node : tree )
    {
        writeDecisionTreeNode( os, node );
    }
}

void storeDecisionTree( const DecisionTree & tree, const std::string & filename )
{
    // Serialize the tree.
    std::ofstream out( filename, std::ios::binary | std::ios::out );
    writeDecisionTree( out, tree );
};

Forest::SharedPointer readForest( std::istream & in )
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

void writeForest( std::ostream & out, const Forest &forest )
{
    // Write the forest header.
    out << 'f';

    // Serialize all trees.
    for ( auto const & tree : forest )
    {
        writeDecisionTree( out, *tree );
    }
}

Forest::SharedPointer loadForest( const std::string &filename )
{
    // Open an input stream for the file.
    std::ifstream in( filename, std::ios::binary | std::ios::in );
    if ( !in.good() ) throw ClientError( std::string( "Can't read file '" ) + filename + "'." );

    // Read the forest.
    return readForest( in );
}

void storeForest( const Forest &forest, const std::string &filename )
{
    // Open an output stream for the file.
    std::ofstream out( filename, std::ios::binary | std::ios::out );
    if ( !out.good() ) throw SupplierError( std::string( "Can't create file '" ) + filename + "'." );

    // Write the forest.
    writeForest( out, forest );
}
