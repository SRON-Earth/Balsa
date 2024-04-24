#ifndef DECISIONTREECLASSIFIER_H
#define DECISIONTREECLASSIFIER_H

#include <algorithm>
#include <iterator>
#include <numeric>

#include "classifier.h"
#include "datatypes.h"
#include "exceptions.h"
#include "iteratortools.h"
#include "serdes.h"

namespace balsa
{

/**
 * A Classifier based on an internal decision tree.
 */
template <typename FeatureIterator, typename OutputIterator>
class DecisionTreeClassifier: public Classifier<FeatureIterator, OutputIterator>
{
public:

    using typename Classifier<FeatureIterator, OutputIterator>::VoteTable;

    typedef std::shared_ptr<DecisionTreeClassifier>       SharedPointer;
    typedef std::shared_ptr<const DecisionTreeClassifier> ConstSharedPointer;

    typedef std::remove_cv_t<typename iterator_value_type<FeatureIterator>::type> FeatureType;
    typedef std::remove_cv_t<typename iterator_value_type<OutputIterator>::type>  LabelType;

    static_assert( std::is_arithmetic<FeatureType>::value, "Feature type should be an integral or floating point type." );
    static_assert( std::is_same<LabelType, Label>::value, "Label type should an unsigned, 8 bits wide, integral type." );

    /**
     * Returns the number of classes distinguished by the classifier.
     */
    unsigned int getClassCount() const
    {
        return m_classCount;
    }

    /**
     * Bulk-classifies a sequence of data points.
     */
    void classify( FeatureIterator pointsStart, FeatureIterator pointsEnd, unsigned int featureCount, OutputIterator labelsStart ) const
    {
        // Check the dimensions of the input data.
        if ( featureCount == 0 ) throw ClientError( "Data points must have at least one feature." );
        auto entryCount = std::distance( pointsStart, pointsEnd );
        if ( entryCount % featureCount ) throw ClientError( "Malformed dataset." );
        if ( featureCount < m_featureCount ) throw ClientError( "Dataset contains too few features." );

        // Determine the number of points in the input data.
        auto pointCount = entryCount / featureCount;

        // Create a table for the label votes.
        VoteTable voteCounts( pointCount, m_classCount );

        // Bulk-classify all points.
        classifyAndVote( pointsStart, pointsEnd, featureCount, voteCounts );

        // Generate the labels.
        for ( unsigned int point = 0; point < pointCount; ++point )
            *labelsStart++ = static_cast<LabelType>( voteCounts.getColumnOfRowMaximum( point ) );
    }

    /**
     * Bulk-classifies a set of points, adding a vote (+1) to the vote table for
     * each point of which the label is 'true'.
     * \param pointsStart An iterator that points to the first feature value of
     *  the first point.
     * \param pointsEnd An itetartor that points to the end of the block of
     *  point data.
     * \param featureCount The number of features for each data point.
     * \param table A table for counting votes.
     * \pre The column count of the vote table must match the number of
     *  features, the row count must match the number of points.
     */
    unsigned int classifyAndVote( FeatureIterator pointsStart, FeatureIterator pointsEnd, unsigned int featureCount, VoteTable & table ) const
    {
        // Check the dimensions of the input data.
        if ( featureCount == 0 ) throw ClientError( "Data points must have at least one feature." );
        auto entryCount = std::distance( pointsStart, pointsEnd );
        if ( entryCount % featureCount ) throw ClientError( "Malformed dataset." );
        if ( featureCount < m_featureCount ) throw ClientError( "Dataset contains too few features." );

        // Determine the number of points in the input data.
        auto pointCount = entryCount / featureCount;

        // Create a list containing all datapoint IDs (0, 1, 2, etc.).
        std::vector<DataPointID> pointIDs( pointCount );
        std::iota( pointIDs.begin(), pointIDs.end(), 0 );

        // Recursively partition the list of point IDs according to the interior node criteria, and classify them by the leaf node labels.
        recursiveClassifyVote( pointIDs.begin(), pointIDs.end(), pointsStart, featureCount, table, NodeID( 0 ) );

        // Return the number of classifiers that voted.
        return 1;
    }

    /**
     * Deserialize a classifier instance from a binary input stream.
     */
    static SharedPointer deserialize( std::istream & is )
    {
        // Create an empty classifier.
        SharedPointer classifier( new DecisionTreeClassifier() );

        // Read the header.
        assert( is.good() );
        expect( is, "tree", "Missing tree header." );
        expect( is, "ccnt", "Missing class count field." );
        classifier->m_classCount = balsa::deserialize<uint32_t>( is );
        expect( is, "fcnt", "Missing feature count field." );
        classifier->m_featureCount = balsa::deserialize<uint32_t>( is );

        // Deserialize the tables.
        is >> classifier->m_leftChildID;
        is >> classifier->m_rightChildID;
        is >> classifier->m_splitFeatureID;
        is >> classifier->m_splitValue;
        is >> classifier->m_label;

        return classifier;
    }

private:

    DecisionTreeClassifier():
    Classifier<FeatureIterator, OutputIterator>(),
    m_leftChildID(0),
    m_rightChildID(0),
    m_splitFeatureID(0),
    m_splitValue(0),
    m_label(0)
    {
    }

    void recursiveClassifyVote( std::vector<DataPointID>::iterator pointIDsStart, std::vector<DataPointID>::iterator pointIDsEnd, FeatureIterator pointsStart, unsigned int featureCount, VoteTable & voteTable, NodeID currentNodeID ) const
    {
        // If the current node is an interior node, split the points along the split value, and classify both halves.
        if ( m_leftChildID( currentNodeID, 0 ) > 0 )
        {
            // Extract the split limit and split dimension of this node.
            auto splitValue = m_splitValue( currentNodeID, 0 );
            auto featureID  = m_splitFeatureID( currentNodeID, 0 );

            // Split the point IDs in two halves: points that lie below the split value, and points that lie on or above the feature split value.
            auto pointIsBelowLimit = [&pointsStart, featureCount, splitValue, featureID]( const unsigned int & pointID )
            {
                return pointsStart[featureCount * pointID + featureID] < splitValue;
            };
            auto secondHalf = std::partition( pointIDsStart, pointIDsEnd, pointIsBelowLimit );

            // Recursively classify-vote both halves.
            recursiveClassifyVote( pointIDsStart, secondHalf, pointsStart, featureCount, voteTable, m_leftChildID( currentNodeID, 0 ) );
            recursiveClassifyVote( secondHalf, pointIDsEnd, pointsStart, featureCount, voteTable, m_rightChildID( currentNodeID, 0 ) );
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

    unsigned int       m_classCount;
    unsigned int       m_featureCount;
    Table<NodeID>      m_leftChildID;
    Table<NodeID>      m_rightChildID;
    Table<FeatureID>   m_splitFeatureID;
    Table<FeatureType> m_splitValue;
    Table<Label>       m_label;
};

} // namespace balsa

#endif // DECISIONTREECLASSIFIER_H
