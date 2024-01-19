#ifndef DECISIONTREECLASSIFIER_H
#define DECISIONTREECLASSIFIER_H

#include "classifier.h"
#include "decisiontrees.h"
#include "exceptions.h"

#include <iostream>

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
 * Class that represents a collection of decision trees that can be iterated.
 */
template<typename FeatureIterator, typename OutputIterator>
class DecisionTreeClassifier: public Classifier<FeatureIterator, OutputIterator>
{
public:
  using typename Classifier<FeatureIterator, OutputIterator>::VoteTable;
  using Classifier<FeatureIterator, OutputIterator>::getFeatureCount;

  typedef std::shared_ptr<DecisionTreeClassifier> SharedPointer;
  typedef std::shared_ptr<const DecisionTreeClassifier> ConstSharedPointer;

  typedef typename std::iterator_traits<FeatureIterator>::value_type FeatureValueType;
  // typedef typename std::iterator_traits<OutputIterator>::value_type LabelValueType;
  typedef unsigned char LabelValueType;

  static typename Classifier<FeatureIterator, OutputIterator>::SharedPointer read( std::ifstream & in )
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
    typedef DecisionTreeClassifier<FeatureIterator, OutputIterator> DecisionTreeClassifierType;
    typename DecisionTreeClassifierType::SharedPointer tree( new DecisionTreeClassifierType( featureCount ) );

    // Read the node count.
    if ( in.eof() ) throw ParseError( "Unexpected end of file." );
    std::size_t nodeCount = readPODValue<std::uint64_t>( in );

    std::cout << "features: " << featureCount << " nodes: " << nodeCount << std::endl;

    tree->m_leftChildID   .resize( nodeCount );
    tree->m_rightChildID  .resize( nodeCount );
    tree->m_splitFeatureID.resize( nodeCount );
    tree->m_splitValue    .resize( nodeCount );
    tree->m_label         .resize( nodeCount );

    in.read( reinterpret_cast<char*>( tree->m_leftChildID   .data() ), nodeCount * sizeof( std::uint32_t ) );
    in.read( reinterpret_cast<char*>( tree->m_rightChildID  .data() ), nodeCount * sizeof( std::uint32_t ) );
    in.read( reinterpret_cast<char*>( tree->m_splitFeatureID.data() ), nodeCount * sizeof( std::uint8_t ) );
    in.read( reinterpret_cast<char*>( tree->m_splitValue    .data() ), nodeCount * sizeof( FeatureValueType ) );
    in.read( reinterpret_cast<char*>( tree->m_label         .data() ), nodeCount * sizeof( LabelValueType ) );

    return tree;
  }

  /**
   * Creates an empty decision tree classifier.
   */
  explicit DecisionTreeClassifier( unsigned int featureCount )
  : Classifier<FeatureIterator, OutputIterator>( featureCount )
  {
  }

  /**
   * Creates a decision tree classifier from a decision tree.
   */
  explicit DecisionTreeClassifier( const DecisionTree<> & decisionTree );

  void classify( FeatureIterator pointsStart, FeatureIterator pointsEnd, OutputIterator labels ) const;

  unsigned int classifyAndVote( FeatureIterator pointsStart, FeatureIterator pointsEnd, VoteTable &table ) const;

private:

  void recursiveClassifyVote( std::vector<DataPointID>::iterator pointIDsStart,
    std::vector<DataPointID>::iterator pointIDsEnd, FeatureIterator pointsStart, VoteTable & voteTable,
    NodeID currentNodeID ) const;

  std::vector<unsigned int    >  m_leftChildID   ;
  std::vector<unsigned int    >  m_rightChildID  ;
  std::vector<unsigned char   >  m_splitFeatureID;
  std::vector<FeatureValueType>  m_splitValue    ;
  std::vector<LabelValueType  >  m_label         ;
};

template <typename FeatureIterator, typename OutputIterator>
DecisionTreeClassifier<FeatureIterator, OutputIterator>::DecisionTreeClassifier( const DecisionTree<> & tree )
: Classifier<FeatureIterator, OutputIterator>( tree.getFeatureCount() )
{
    // Pre-allocate internal data structures.
    auto nodeCount = tree.getNodeCount();
    m_leftChildID   .reserve( nodeCount );
    m_rightChildID  .reserve( nodeCount );
    m_splitFeatureID.reserve( nodeCount );
    m_splitValue    .reserve( nodeCount );
    m_label         .reserve( nodeCount );

    // Convert tree representation.
    for ( auto const & node : tree )
    {
        m_leftChildID   .push_back( node.leftChildID    );
        m_rightChildID  .push_back( node.rightChildID   );
        m_splitFeatureID.push_back( node.splitFeatureID );
        m_splitValue    .push_back( node.splitValue     );
        m_label         .push_back( node.label          );
    }
}

template <typename FeatureIterator, typename OutputIterator>
void DecisionTreeClassifier<FeatureIterator, OutputIterator>::classify( FeatureIterator pointsStart, FeatureIterator pointsEnd,
    OutputIterator labels ) const
{
    // Check the dimensions of the input data.
    auto rawFeatureCount = std::distance( pointsStart, pointsEnd );
    auto featureCount    = getFeatureCount();
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
unsigned int DecisionTreeClassifier<FeatureIterator, OutputIterator>::classifyAndVote( FeatureIterator pointsStart, FeatureIterator pointsEnd,
    VoteTable & table ) const
{
    // Check the dimensions of the input data.
    auto rawFeatureCount = std::distance( pointsStart, pointsEnd );
    auto featureCount    = getFeatureCount();
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

    // Return the number of classifiers that voted.
    return 1;
}

template <typename FeatureIterator, typename OutputIterator>
void DecisionTreeClassifier<FeatureIterator, OutputIterator>::recursiveClassifyVote( std::vector<DataPointID>::iterator pointIDsStart,
    std::vector<DataPointID>::iterator pointIDsEnd, FeatureIterator pointsStart, VoteTable & voteTable,
    NodeID currentNodeID ) const
{
    // If the current node is an interior node, split the points along the split value, and classify both halves.
    if ( m_leftChildID[currentNodeID] > 0 )
    {
        // Extract the split limit and split dimension of this node.
        auto splitValue = m_splitValue[currentNodeID];
        auto featureID  = m_splitFeatureID[currentNodeID];

        // Retrieve feature count.
        auto featureCount = getFeatureCount();

        // Split the point IDs in two halves: points that lie below and points that lie on or above the feature split value.
        auto pointIsBelowLimit = [&pointsStart,featureCount,splitValue,featureID]( const unsigned int &pointID )
        {
            return pointsStart[featureCount * pointID + featureID] < splitValue;
        };
        auto secondHalf = std::partition( pointIDsStart, pointIDsEnd, pointIsBelowLimit );

        // Recursively classify both halves.
        recursiveClassifyVote( pointIDsStart, secondHalf, pointsStart, voteTable, m_leftChildID [currentNodeID] );
        recursiveClassifyVote( secondHalf , pointIDsEnd , pointsStart, voteTable, m_rightChildID[currentNodeID] );
    }

    // If the current node is a leaf node (and the label is 'true'), increment the true count in the output.
    else
    {
        if ( m_label[currentNodeID] )
        {
            for ( auto it( pointIDsStart ), end( pointIDsEnd ); it != end; ++it )
            {
                assert( *it < voteTable.size() );
                ++voteTable[*it];
            }
        }
    }
}

#endif // DECISIONTREECLASSIFIER_H
