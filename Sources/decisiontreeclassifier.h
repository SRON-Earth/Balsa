#ifndef DECISIONTREECLASSIFIER_H
#define DECISIONTREECLASSIFIER_H

#include <iterator>
#include <numeric>
#include <algorithm>

#include "classifier.h"

#include "exceptions.h"
#include "serdes.h"

/**
 * A Classifier based on an internal decision tree.
 */
template <typename FeatureIterator, typename OutputIterator, typename FeatureType = typename std::iterator_traits<FeatureIterator>::value_type, typename LabelType = typename std::iterator_traits<OutputIterator>::value_type>
class DecisionTreeClassifier: public Classifier<FeatureIterator, OutputIterator>
{
    static_assert( std::is_arithmetic<FeatureType>::value, "Feature type should be an integral or floating point type." );
    static_assert( std::is_integral<LabelType>::value, "Label type should be integral." );

public:

    using typename Classifier<FeatureIterator, OutputIterator>::VoteTable;
    using Classifier<FeatureIterator, OutputIterator>::getFeatureCount;

    typedef std::shared_ptr<DecisionTreeClassifier>       SharedPointer;
    typedef std::shared_ptr<const DecisionTreeClassifier> ConstSharedPointer;

    /**
     * Deserialize a classifier instance from a binary input stream.
     */
    static SharedPointer deserialize( std::istream & is );

    void classify( FeatureIterator pointsStart, FeatureIterator pointsEnd, OutputIterator labels ) const;

    unsigned int classifyAndVote( FeatureIterator pointsStart, FeatureIterator pointsEnd, VoteTable & table ) const;

private:

    explicit DecisionTreeClassifier( unsigned int featureCount );

    void recursiveClassifyVote( std::vector<DataPointID>::iterator pointIDsStart, std::vector<DataPointID>::iterator pointIDsEnd, FeatureIterator pointsStart, VoteTable & voteTable, NodeID currentNodeID ) const;

    Table<NodeID>        m_leftChildID;
    Table<NodeID>        m_rightChildID;
    Table<FeatureID>     m_splitFeatureID;
    Table<FeatureType>   m_splitValue;
    Table<unsigned char> m_label;
};

template <typename FeatureIterator, typename OutputIterator, typename FeatureType, typename LabelType>
typename DecisionTreeClassifier<FeatureIterator, OutputIterator, FeatureType, LabelType>::SharedPointer DecisionTreeClassifier<FeatureIterator, OutputIterator, FeatureType, LabelType>::deserialize( std::istream & is )
{
    // Read the header.
    assert( is.good() );
    expect( is, "tree", "Missing tree header." );
    expect( is, "fcnt", "Missing feature count field." );
    auto featureCount = ::deserialize<uint32_t>( is );

    // Create an empty classifier.
    SharedPointer classifier( new DecisionTreeClassifier( featureCount ) );

    // Deserialize the tables.
    is >> classifier->m_leftChildID;
    is >> classifier->m_rightChildID;
    is >> classifier->m_splitFeatureID;
    is >> classifier->m_splitValue;
    is >> classifier->m_label;

    return classifier;
}

template <typename FeatureIterator, typename OutputIterator, typename FeatureType, typename LabelType>
DecisionTreeClassifier<FeatureIterator, OutputIterator, FeatureType, LabelType>::DecisionTreeClassifier( unsigned int featureCount ):
Classifier<FeatureIterator, OutputIterator>( featureCount ),
m_leftChildID(featureCount),
m_rightChildID(featureCount),
m_splitFeatureID(featureCount),
m_splitValue(featureCount),
m_label(featureCount)
{
}

template <typename FeatureIterator, typename OutputIterator, typename FeatureType, typename LabelType>
void DecisionTreeClassifier<FeatureIterator, OutputIterator, FeatureType, LabelType>::classify( FeatureIterator pointsStart, FeatureIterator pointsEnd, OutputIterator labels ) const
{
    // Check the dimensions of the input data.
    auto rawFeatureCount = std::distance( pointsStart, pointsEnd );
    auto featureCount    = getFeatureCount();
    assert( rawFeatureCount > 0 );
    assert( ( rawFeatureCount % featureCount ) == 0 );

    // Create a table for the label votes.
    unsigned int pointCount = rawFeatureCount / featureCount;
    VoteTable    voteCounts( pointCount, featureCount );

    // Bulk-classify all points.
    classifyAndVote( pointsStart, pointsEnd, voteCounts );

    // Generate the labels.
    for ( unsigned int point = 0; point < pointCount; ++point ) *labels++ = static_cast<LabelType>( voteCounts.getColumnOfRowMaximum( point ) );
}

template <typename FeatureIterator, typename OutputIterator, typename FeatureType, typename LabelType>
unsigned int DecisionTreeClassifier<FeatureIterator, OutputIterator, FeatureType, LabelType>::classifyAndVote( FeatureIterator pointsStart, FeatureIterator pointsEnd, VoteTable & table ) const
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

    // Recursively partition the list of point IDs according to the interior node criteria, and classify them by the leaf node labels.
    recursiveClassifyVote( pointIDs.begin(), pointIDs.end(), pointsStart, table, NodeID( 0 ) );

    // Return the number of classifiers that voted.
    return 1;
}

template <typename FeatureIterator, typename OutputIterator, typename FeatureType, typename LabelType>
void DecisionTreeClassifier<FeatureIterator, OutputIterator, FeatureType, LabelType>::recursiveClassifyVote( std::vector<DataPointID>::iterator pointIDsStart, std::vector<DataPointID>::iterator pointIDsEnd, FeatureIterator pointsStart, VoteTable & voteTable, NodeID currentNodeID ) const
{
    // If the current node is an interior node, split the points along the split value, and classify both halves.
    if ( m_leftChildID(currentNodeID, 0) > 0 )
    {
        // Extract the split limit and split dimension of this node.
        auto splitValue = m_splitValue( currentNodeID, 0 );
        auto featureID  = m_splitFeatureID( currentNodeID, 0 );

        // Retrieve feature count.
        auto featureCount = getFeatureCount();

        // Split the point IDs in two halves: points that lie below the split value, and points that lie on or above the feature split value.
        auto pointIsBelowLimit = [&pointsStart, featureCount, splitValue, featureID]( const unsigned int & pointID )
        {
            return pointsStart[featureCount * pointID + featureID] < splitValue;
        };
        auto secondHalf = std::partition( pointIDsStart, pointIDsEnd, pointIsBelowLimit );

        // Recursively classify-vote both halves.
        recursiveClassifyVote( pointIDsStart, secondHalf, pointsStart, voteTable, m_leftChildID(currentNodeID, 0) );
        recursiveClassifyVote( secondHalf, pointIDsEnd, pointsStart, voteTable, m_rightChildID(currentNodeID, 0) );
    }

    // If the current node is a leaf node, cast a vote for the node-label for each point.
    else
    {
        auto label = m_label( currentNodeID, 0 );
        for ( auto it( pointIDsStart ), end( pointIDsEnd ); it != end; ++it )
        {
            ++voteTable( *it, label );
        }
    }
}

#endif // DECISIONTREECLASSIFIER_H
