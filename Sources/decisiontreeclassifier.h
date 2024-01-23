#ifndef DECISIONTREECLASSIFIER_H
#define DECISIONTREECLASSIFIER_H

#include <iterator>
#include <numeric>

#include "classifier.h"
#include "decisiontrees.h"
#include "exceptions.h"
#include "serdes.h"

/**
 * Class that represents a collection of decision trees that can be iterated.
 */
template <typename FeatureIterator,
    typename OutputIterator,
    typename FeatureType = std::iterator_traits<FeatureIterator>::value_type,
    typename LabelType   = std::iterator_traits<OutputIterator>::value_type>
class DecisionTreeClassifier: public Classifier<FeatureIterator, OutputIterator>
{
    static_assert( std::is_arithmetic<FeatureType>::value,
        "Feature type should be an integral or floating point type." );
    static_assert( std::is_same<LabelType, bool>::value, "Label type should be 'bool'." );

public:

    using typename Classifier<FeatureIterator, OutputIterator>::VoteTable;
    using Classifier<FeatureIterator, OutputIterator>::getFeatureCount;

    typedef std::shared_ptr<DecisionTreeClassifier> SharedPointer;
    typedef std::shared_ptr<const DecisionTreeClassifier> ConstSharedPointer;

    /**
     * Deserialize a classifier instance from a binary input stream.
     */
    static SharedPointer deserialize( std::istream & is );

    /**
     * Creates a decision tree classifier from a decision tree.
     */
    explicit DecisionTreeClassifier( unsigned int featureCount, const DecisionTree<> & decisionTree );

    void classify( FeatureIterator pointsStart, FeatureIterator pointsEnd, OutputIterator labels ) const;

    unsigned int classifyAndVote( FeatureIterator pointsStart, FeatureIterator pointsEnd, VoteTable & table ) const;

private:

    explicit DecisionTreeClassifier( unsigned int featureCount );

    void recursiveClassifyVote( std::vector<DataPointID>::iterator pointIDsStart,
        std::vector<DataPointID>::iterator pointIDsEnd,
        FeatureIterator pointsStart,
        VoteTable & voteTable,
        NodeID currentNodeID ) const;

    std::vector<NodeID> m_leftChildID;
    std::vector<NodeID> m_rightChildID;
    std::vector<FeatureID> m_splitFeatureID;
    std::vector<FeatureType> m_splitValue;
    std::vector<unsigned char> m_label;
};

template <typename FeatureIterator, typename OutputIterator, typename FeatureType, typename LabelType>
typename DecisionTreeClassifier<FeatureIterator, OutputIterator, FeatureType, LabelType>::SharedPointer
DecisionTreeClassifier<FeatureIterator, OutputIterator, FeatureType, LabelType>::deserialize( std::istream & is )
{
    // Read the header.
    assert( is.good() );
    if ( is.eof() ) throw ParseError( "Unexpected end of file." );
    char marker = ::deserialize<char>( is );
    if ( is.eof() ) throw ParseError( "Unexpected end of file." );
    if ( marker != 't' ) throw ParseError( "Unexpected header block." );

    // Read the feature count.
    if ( is.eof() ) throw ParseError( "Unexpected end of file." );
    unsigned int featureCount = ::deserialize<std::uint8_t>( is );

    // Read the node count.
    if ( is.eof() ) throw ParseError( "Unexpected end of file." );
    std::size_t nodeCount = ::deserialize<std::uint64_t>( is );

    // Create an empty decision tree classifier.
    SharedPointer classifier( new DecisionTreeClassifier( featureCount ) );

    // Allocate space for decision tree nodes.
    classifier->m_leftChildID.resize( nodeCount );
    classifier->m_rightChildID.resize( nodeCount );
    classifier->m_splitFeatureID.resize( nodeCount );
    classifier->m_splitValue.resize( nodeCount );
    classifier->m_label.resize( nodeCount );

    // Deserialize decision tree nodes.
    is.read( reinterpret_cast<char *>( classifier->m_leftChildID.data() ), nodeCount * sizeof( std::uint32_t ) );
    is.read( reinterpret_cast<char *>( classifier->m_rightChildID.data() ), nodeCount * sizeof( std::uint32_t ) );
    is.read( reinterpret_cast<char *>( classifier->m_splitFeatureID.data() ), nodeCount * sizeof( std::uint8_t ) );
    is.read( reinterpret_cast<char *>( classifier->m_splitValue.data() ), nodeCount * sizeof( FeatureType ) );
    is.read( reinterpret_cast<char *>( classifier->m_label.data() ), nodeCount * sizeof( LabelType ) );

    return classifier;
}

template <typename FeatureIterator, typename OutputIterator, typename FeatureType, typename LabelType>
DecisionTreeClassifier<FeatureIterator, OutputIterator, FeatureType, LabelType>::DecisionTreeClassifier(
    unsigned int featureCount )
: Classifier<FeatureIterator, OutputIterator>( featureCount )
{
}

template <typename FeatureIterator, typename OutputIterator, typename FeatureType, typename LabelType>
DecisionTreeClassifier<FeatureIterator, OutputIterator, FeatureType, LabelType>::DecisionTreeClassifier(
    unsigned int featureCount,
    const DecisionTree<> & tree )
: Classifier<FeatureIterator, OutputIterator>( featureCount )
{
    // Pre-allocate internal data structures.
    auto nodeCount = tree.getNodeCount();
    m_leftChildID.reserve( nodeCount );
    m_rightChildID.reserve( nodeCount );
    m_splitFeatureID.reserve( nodeCount );
    m_splitValue.reserve( nodeCount );
    m_label.reserve( nodeCount );

    // Convert tree representation.
    for ( const auto & node : tree )
    {
        m_leftChildID.push_back( node.leftChildID );
        m_rightChildID.push_back( node.rightChildID );
        m_splitFeatureID.push_back( node.splitFeatureID );
        m_splitValue.push_back( node.splitValue );
        m_label.push_back( node.label );
    }
}

template <typename FeatureIterator, typename OutputIterator, typename FeatureType, typename LabelType>
void DecisionTreeClassifier<FeatureIterator, OutputIterator, FeatureType, LabelType>::classify(
    FeatureIterator pointsStart,
    FeatureIterator pointsEnd,
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

template <typename FeatureIterator, typename OutputIterator, typename FeatureType, typename LabelType>
unsigned int DecisionTreeClassifier<FeatureIterator, OutputIterator, FeatureType, LabelType>::classifyAndVote(
    FeatureIterator pointsStart,
    FeatureIterator pointsEnd,
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

template <typename FeatureIterator, typename OutputIterator, typename FeatureType, typename LabelType>
void DecisionTreeClassifier<FeatureIterator, OutputIterator, FeatureType, LabelType>::recursiveClassifyVote(
    std::vector<DataPointID>::iterator pointIDsStart,
    std::vector<DataPointID>::iterator pointIDsEnd,
    FeatureIterator pointsStart,
    VoteTable & voteTable,
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

        // Split the point IDs in two halves: points that lie below and points that lie on or above the feature split
        // value.
        auto pointIsBelowLimit = [&pointsStart, featureCount, splitValue, featureID]( const unsigned int & pointID )
        {
            return pointsStart[featureCount * pointID + featureID] < splitValue;
        };
        auto secondHalf = std::partition( pointIDsStart, pointIDsEnd, pointIsBelowLimit );

        // Recursively classify both halves.
        recursiveClassifyVote( pointIDsStart, secondHalf, pointsStart, voteTable, m_leftChildID[currentNodeID] );
        recursiveClassifyVote( secondHalf, pointIDsEnd, pointsStart, voteTable, m_rightChildID[currentNodeID] );
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
