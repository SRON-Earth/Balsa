#ifndef INDEXEDDECISIONTREE_H
#define INDEXEDDECISIONTREE_H

#include <deque>
#include <fstream>
#include <valarray>
#include <vector>

#include "datatools.h"
#include "datatypes.h"
#include "table.h"
#include "weightedcoin.h"

/**
 * A decision tree with an internal search index for fast training.
 */
template <typename FeatureType = double>
class IndexedDecisionTree
{

    // Forward declarations.
    class FeatureIndexEntry;
    class Node;

public:

    typedef std::shared_ptr<IndexedDecisionTree> SharedPointer;

    typedef WeightedCoin<> WeightedCoinType;
    typedef WeightedCoinType::ValueType SeedType;

    /**
     * Creates an indexed decision tree with one root node from scratch.
     * N.B. this is an expensive operation, because construction builds sorted
     * indices. When training multiple trees on the same data, it is much more
     * efficient to create one tree and to copy the initial tree multiple times.
     */
    IndexedDecisionTree( const Table<FeatureType> & dataPoints, const Table<Label> & labels, unsigned int featuresToConsider, unsigned int maximumDistanceToRoot = std::numeric_limits<unsigned int>::max() ):
    m_dataPoints( dataPoints ),
    m_labels( labels ),
    m_featuresToConsider( featuresToConsider ),
    m_maximumDistanceToRoot( maximumDistanceToRoot ),
    m_impurityThreshold( 0 ) // Between 0 and 0.5. A value of 0 means any split that is an improvement will be made, 0.5 means no splits are made.
    {
        // Determine the number of points and features in the dataset.
        auto pointCount   = dataPoints.getRowCount();
        auto featureCount = dataPoints.getColumnCount();
        auto labelCount   = labels.getRowCount();
        if ( labelCount != pointCount ) throw ClientError( "The number of points in the training set doesn't match the number of labels." );
        assert( featuresToConsider > 0 && featuresToConsider <= featureCount );

        // Check preconditions.
        assert( pointCount == labels.getRowCount() );
        assert( labels.getColumnCount() == 1 );

        // Build a sorted point index for each feature.
        for ( FeatureID feature = 0; feature < featureCount; ++feature )
        {
            // Create an empty index for this feature with enough capacity for one entry per data point.
            m_featureIndex.push_back( SingleFeatureIndex() );
            auto & singleFeatureIndex = *m_featureIndex.rbegin();
            singleFeatureIndex.reserve( dataPoints.getRowCount() );

            // Add all the data points to the single-feature index.
            for ( DataPointID point = 0; point < pointCount; ++point )
            {
                singleFeatureIndex.push_back( FeatureIndexEntry( dataPoints( point, feature ), point, labels( point, 0 ) ) );
            }

            // Sort the index by feature value.
            std::sort( singleFeatureIndex.begin(), singleFeatureIndex.end() );
        }

        // Create a frequency table for all labels in the data set.
        LabelFrequencyTable labelCounts( m_labels.begin(), m_labels.end() );
        assert( labelCount == labelCounts.getTotal() );
        assert( labelCounts.invariant() );

        // Create the root node (it contains all points).
        m_nodes.push_back( Node( labelCounts, 0, 0 ) );

        // If the root node is still growable, add it to the list of growable nodes.
        if ( isGrowableNode( 0 ) ) m_growableLeaves.push_back( 0 );
    }

    /**
     * Reinitialize the state of the random engine used to select features to
     * consider when deciding where to split.
     */
    void seed( SeedType value )
    {
        m_coin.seed( value );
    }

    /**
     * Grows the entire tree until no more progress is possible.
     */
    void grow()
    {
        while ( isGrowable() ) growNextLeaf();
    }

    /**
     * Returns true iff there are any growable nodes left in the tree.
     */
    bool isGrowable() const
    {
        return m_growableLeaves.size() > 0;
    }

    /**
     * Grows one of the remaining growable leaves.
     * \pre isGrowable()
     */
    void growNextLeaf()
    {
        // Check precondition.
        assert( isGrowable() );

        // Grow the next growable leaf.
        auto leaf = m_growableLeaves.front();
        m_growableLeaves.pop_front();
        growLeaf( leaf );
    }

    /**
     * Write the tree model to a Dotty file, suitable for visualization
     */
    void writeGraphviz( const std::string & filename ) const
    {
        // Create the file.
        std::ofstream out;
        out.open( filename );
        if ( !out.good() ) throw SupplierError( "Could not open file for writing." );

        // Write the graph data.
        out << "digraph G" << std::endl;
        out << "{" << std::endl;
        for ( NodeID nodeID = 0; nodeID < m_nodes.size(); ++nodeID )
        {
            // Write the node label.
            auto &node = m_nodes[nodeID];
            std::stringstream info;
            info << 'N' << nodeID << " = " << static_cast<int>( node.getLabel() ) << " counts: " << node.getLabelCounts().asText();
            out << "    node" << nodeID << "[shape=box label=\"" << info.str() << "\"]" << std::endl;

            // Write the links to the children.
            if ( !node.isLeafNode() )
            {
                auto splitFeature = node.getSplit().getFeatureID();
                auto splitValue   = node.getSplit().getFeatureValue();
                out << "    node" << nodeID << " -> " << "node" << node.getLeftChild()  << " [label=\"F" << static_cast<int>( splitFeature ) << " < " << splitValue << "\"];" << std::endl;
                out << "    node" << nodeID << " -> " << "node" << node.getRightChild() << ';' << std::endl;
            }
        }
        out << "}" << std::endl;

        // Close the file.
        out.close();
    }

    /**
     * Serialize this indexed decision tree as a plain, un-indexed decision tree classifier.
     */
    void writeDecisionTreeClassifier( std::ostream &binOut )
    {
        // Create data structures that directly mirror the internal table-representation used by the classifier.
        NodeID nodeCount = m_nodes.size();
        Table<NodeID>        leftChildID    ( nodeCount, 1, 0 );
        Table<NodeID>        rightChildID   ( nodeCount, 1, 0 );
        Table<FeatureID>     splitFeatureID ( nodeCount, 1, 0 );
        Table<FeatureType>   splitValue     ( nodeCount, 1, 0 );
        Table<unsigned char> label          ( nodeCount, 1, 0 );

        // Copy the tree data to the tables.
        for ( NodeID nodeID = 0; nodeID < nodeCount; ++nodeID )
        {
            auto &node = m_nodes[nodeID];
            auto &split = node.getSplit();
            leftChildID   ( nodeID, 0 ) = node.getLeftChild();
            rightChildID  ( nodeID, 0 ) = node.getRightChild();
            splitFeatureID( nodeID, 0 ) = split.getFeatureID();
            splitValue    ( nodeID, 0 ) = split.getFeatureValue();
            label         ( nodeID, 0 ) = node.getLabel();
        }

        // Write the header and the data tables of the classifier.
        binOut.write( "tree", 4 );
        binOut.write( "fcnt", 4 );
        serialize( binOut, static_cast<uint32_t>( m_dataPoints.getColumnCount() ) );
        leftChildID.serialize( binOut );
        rightChildID.serialize( binOut );
        splitFeatureID.serialize( binOut );
        splitValue.serialize( binOut );
        label.serialize( binOut );
    }

private:

    /**
     * A floating-point type used to calculate the information gain of splits.
     */
    typedef FeatureType ImpurityType;

    /**
     * A list of points and labels, sorted by one particular feature.
     */
    typedef std::vector<FeatureIndexEntry> SingleFeatureIndex;

    /**
     * The combination of a Split (i.e. the separation of a set of points along one feature axis) and the label frequency tables
     * of the left- and right half, that would result after the split.
     */
    class SplitCandidate: public Split<FeatureType>
    {
    public:

        /**
         * Constructs an invalid split.
         * Invalid splits have an impurity greater than 1. Any real split would have a lower impurity.
         */
        SplitCandidate():
        m_leftCounts( 0 ),
        m_rightCounts( 0 ),
        m_impurity( std::numeric_limits<FeatureType>::max() )
        {
        }

        /**
         * Constructor.
         * \param Split a possible separation between two sets of labeled data points.
         * \param leftCounts The counts of the various labels of points to the left of the split.
         * \param rightCounts The counts of the various labels of points to the right of the split.
         */
        SplitCandidate( const Split<FeatureType> & split, const LabelFrequencyTable & leftCounts, const LabelFrequencyTable & rightCounts ):
        m_split( split ),
        m_leftCounts( leftCounts ),
        m_rightCounts( rightCounts )
        {
            // Calculate the post-split impurity.
            auto leftCount     = leftCounts.getTotal();
            auto rightCount    = rightCounts.getTotal();
            auto totalCount    = leftCount + rightCount;
            auto leftImpurity  = leftCounts.giniImpurity<ImpurityType>();
            auto rightImpurity = rightCounts.giniImpurity<ImpurityType>();
            m_impurity         = ( leftImpurity * leftCount + rightImpurity * rightCount ) / totalCount;
        }

        /**
         * Returns true iff this candidate represents a valid split.
         */
        bool isValid() const
        {
            return m_impurity <= 1.0;
        }

        const Split<FeatureType> & getSplit() const
        {
            return m_split;
        }

        const LabelFrequencyTable & getLeftCounts() const
        {
            return m_leftCounts;
        }

        const LabelFrequencyTable & getRightCounts() const
        {
            return m_rightCounts;
        }

        const ImpurityType getImpurity() const
        {
            return m_impurity;
        }

    private:

        Split<FeatureType>  m_split;
        LabelFrequencyTable m_leftCounts;
        LabelFrequencyTable m_rightCounts;
        ImpurityType        m_impurity;
    };

    /**
     * Internal representation of a node in the decision tree.
     */
    class Node
    {
    public:

        /**
         * Constructor.
         * \param labelCounts The absolute counts of the points in this node, per label value.
         * \param indexOffset The offset in the sorted feature index tables at which the data of this node can be found.
         * \param distanceToRoot The number of hops to this node from the root node of the tree.
         */
        Node( const LabelFrequencyTable & labelCounts, std::size_t indexOffset, unsigned int distanceToRoot ):
        m_leftChild( 0 ),
        m_rightChild( 0 ),
        m_indexOffset( indexOffset ),
        m_distanceToRoot( distanceToRoot ),
        m_labelCounts( labelCounts ),
        m_label( m_labelCounts.getMostFrequentLabel() )
        {
        }

        /**
         * Returns the offset of the data of this node in the feature indices.
         */
        std::size_t getIndexOffset() const
        {
            return m_indexOffset;
        }

        /**
         * Update the split data in this node.
         * \pre isLeafNode()
         */
        void setSplit( const Split<FeatureType> & split, NodeID leftNodeID, NodeID rightNodeID )
        {
            assert( isLeafNode() );
            m_split      = split;
            m_leftChild  = leftNodeID;
            m_rightChild = rightNodeID;
            assert( ( m_leftChild && m_rightChild ) || !( m_leftChild || m_rightChild ) ); // Both children must be non-null.
        }

        /**
         * Returns true iff this is a leaf node.
         */
        bool isLeafNode() const
        {
            return m_leftChild == 0;
        }

        /**
         * Get the label of this node (the most frequent label).
         */
        Label getLabel() const
        {
            return m_label;
        }

        /**
         * Returns the total number of points in the node.
         */
        std::size_t getPointCount() const
        {
            return m_labelCounts.getTotal();
        }

        /**
         * Returns the table of absolute counts of each label within this node.
         */
        const LabelFrequencyTable & getLabelCounts() const
        {
            return m_labelCounts;
        }

        /**
         * Returns the number of ancestors of this node.
         */
        unsigned int getDistanceToRoot() const
        {
            return m_distanceToRoot;
        }

        /**
         * Returns the node ID of the left child of this node, or 0 for leaf nodes.
         */
        NodeID getLeftChild() const
        {
            return m_leftChild;
        }

        /**
         * Returns the node ID of the right child of this node, or 0 for leaf nodes.
         */
        NodeID getRightChild() const
        {
            return m_rightChild;
        }

        /**
         * Returns the split (only valid for leaf nodes).
         */
        const Split<FeatureType> & getSplit() const
        {
            return m_split;
        }

        /**
         * Returns a text representation of the node, for debugging purposes.
         */
        const std::string getInfo() const
        {
            std::stringstream ss;
            ss << "Children: " << m_leftChild << ' ' << m_rightChild << " Level: " << m_distanceToRoot << " Label counts: " ;
            ss << m_labelCounts.asText();
            return ss.str();
        }

    private:

        NodeID              m_leftChild;
        NodeID              m_rightChild;
        std::size_t         m_indexOffset;
        Split<FeatureType>  m_split;
        unsigned int        m_distanceToRoot;
        LabelFrequencyTable m_labelCounts;
        Label               m_label;
    };

    /**
     * An entry in the internal feature index.
     */
    class FeatureIndexEntry
    {
    public:

        FeatureIndexEntry( FeatureType featureValue, DataPointID pointID, Label label ):
        m_featureValue( featureValue ),
        m_pointID( pointID ),
        m_label( label )
        {
        }

        bool operator<( const FeatureIndexEntry & other ) const
        {
            // Entries are to be ordered by feature value only.
            return m_featureValue < other.m_featureValue;
        }

        FeatureType m_featureValue;
        DataPointID m_pointID;
        Label       m_label;
    };

    /**
     * Apply the specified split to the node.
     * \pre The node must be a leaf node.
     */
    void splitNode( NodeID nodeID, const SplitCandidate & splitCandidate )
    {
        // Check the precondition.
        Node &node = m_nodes[nodeID];
        assert( node.isLeafNode() );

        // Split the feature index.
        std::size_t leftPointCount = splitCandidate.getLeftCounts().getTotal();
        assert( node.isLeafNode() );
        for ( FeatureID featureID = 0; featureID < m_featureIndex.size(); ++featureID )
        {
            // No work is necessary for the feature on which the split is performed.
            auto splitFeature = splitCandidate.getSplit().getFeatureID();
            auto splitValue   = splitCandidate.getSplit().getFeatureValue();
            if ( featureID == splitFeature ) continue;

            // For other features, partition the points in the index along the split edge, but keep them sorted.
            auto nodeDataStart = m_featureIndex[featureID].begin() + node.getIndexOffset();
            auto nodeDataEnd   = nodeDataStart + node.getPointCount();
            auto predicate     = [this, splitFeature, splitValue]( const auto & entry ) -> bool
            {
                return this->m_dataPoints( entry.m_pointID, splitFeature ) < splitValue;
            };

            auto secondNodeData = std::stable_partition( nodeDataStart, nodeDataEnd, predicate );
            assert( secondNodeData != nodeDataStart );

            // Make sure the point count is consistent with what is in the split candidate.
            auto distance = std::distance( nodeDataStart, secondNodeData );
            assert( distance > 0 );
            auto newLeftPointCount = static_cast<std::size_t>( distance );
            assert( newLeftPointCount == leftPointCount );
        }
        assert( node.isLeafNode() );

        // Create the child nodes before adding them to the node table, because that will invalidate the 'node' reference.
        NodeID leftChildID  = m_nodes.size();
        NodeID rightChildID = leftChildID + 1;
        assert( leftPointCount );
        Node leftChild = Node( splitCandidate.getLeftCounts(), node.getIndexOffset(), node.getDistanceToRoot() + 1 );
        Node rightChild = Node( splitCandidate.getRightCounts(), node.getIndexOffset() + splitCandidate.getLeftCounts().getTotal(), node.getDistanceToRoot() + 1 );
        node.setSplit( splitCandidate.getSplit(), leftChildID, rightChildID );

        // Put the created child nodes in the list.
        m_nodes.push_back( leftChild  );
        m_nodes.push_back( rightChild );

        // Add the children to the list of growable nodes, if applicable.
        if ( isGrowableNode( leftChildID ) ) m_growableLeaves.push_back( leftChildID );
        if ( isGrowableNode( rightChildID ) ) m_growableLeaves.push_back( rightChildID );
    }

    /**
     * Find the best possible split for the specified leaf node, taking randomly
     * selected features into account.
     */
    SplitCandidate findBestSplit( NodeID node )
    {
        // Check precondition.
        auto featureCount = m_dataPoints.getColumnCount();
        assert( m_featuresToConsider <= featureCount );

        // Randomly scan the required number of features.
        SplitCandidate bestSplit;
        assert( bestSplit.getImpurity() > m_nodes[node].getLabelCounts().template giniImpurity<ImpurityType>() );
        auto           featuresToScan = m_featuresToConsider;
        std::vector<FeatureID> skippedFeatures;
        for ( FeatureID featureID = 0; featureID < featureCount; ++featureID )
        {
            // Decide whether or not to consider this feature.
            auto featuresLeft        = featureCount - featureID;
            bool considerThisFeature = m_coin.flip( featuresToScan, featuresLeft );
            if ( !considerThisFeature )
            {
                skippedFeatures.push_back( featureID );
                continue;
            }

            // Use up one 'credit'.
            assert( featuresToScan > 0 );
            --featuresToScan;

            // Scan the feature for a split that is better than what was already found.
            bestSplit = findBestSplitForFeature( m_nodes[node], featureID, bestSplit );
        }
        assert( skippedFeatures.size() == featureCount - m_featuresToConsider );

        // If a valid split has been found, return it.
        if ( bestSplit.isValid() ) return bestSplit;

        // Since no valid split was found, scan all features that were initially skipped.
        for ( auto featureID: skippedFeatures )
        {
            // Return the first candidate split.
            bestSplit = findBestSplitForFeature( m_nodes[node], featureID, bestSplit );
            if ( bestSplit.isValid() ) return bestSplit;
        }

        // All points in this node must have exactly the same feature values. Issue a warning.
        // TODO: proper warning object.
        DataPointID anyPointInNode = m_featureIndex[0][ m_nodes[node].getIndexOffset() ].m_pointID;
        std::cout << "WARNING: training data contains a cluster of identical points with different labels." << std::endl;
        std::cout << "Feature values:";
        for ( unsigned int f = 0; f < featureCount; ++f )
        {
            std::cout << ' ' << m_dataPoints( anyPointInNode, f );
        }
        std::cout << std::endl;
        std::cout << "Label frequencies: " << m_nodes[node].getLabelCounts().asText() << std::endl;

        return bestSplit;
    }

    /**
     * Find the best split for a particular node and feature, that is at least as good as the supplied minimal best split.
     * \param node The node that will be examined.
     * \param featureID The feature that will be examined.
     * \return Either returns minimalBestSplit, or, if found, a better split along the specified featureID.
     */
    SplitCandidate findBestSplitForFeature( const Node & node, FeatureID featureID, const SplitCandidate & minimalBestSplit ) const
    {
        // Find the region of the index that covers this node and feature.
        auto begin = m_featureIndex[featureID].begin() + node.getIndexOffset();
        auto end   = begin + node.getPointCount();
        assert( begin != end );

        // Search for a better split than the supplied minimal best split.
        auto                bestSplit        = minimalBestSplit;
        FeatureType         currentBlockValue = begin->m_featureValue;
        LabelFrequencyTable leftSideLabelCounts( node.getLabelCounts().size() );
        LabelFrequencyTable rightSideLabelCounts( node.getLabelCounts() );

        assert( leftSideLabelCounts.invariant() );
        assert( rightSideLabelCounts.invariant() );
        for ( auto it( begin ); it != end; ++it )
        {
            // If this is the end of a block of equal-valued points, test if this split would be an improvement over the current best.
            if ( it->m_featureValue > currentBlockValue )
            {
                SplitCandidate possibleSplit( Split( featureID, it->m_featureValue ), leftSideLabelCounts, rightSideLabelCounts );
                if ( possibleSplit.getImpurity() < bestSplit.getImpurity() )
                {
                    bestSplit = possibleSplit;
                }
            }

            // Move the current block value to the value of the currently visited point.
            currentBlockValue = it->m_featureValue;

            // Update the left- and right-hand label counts as the point is visited.
            leftSideLabelCounts.increment( it->m_label );
            rightSideLabelCounts.decrement( it->m_label );
        }

        return bestSplit;
    }

    void growLeaf( NodeID nodeID )
    {
        assert( m_nodes[nodeID].isLeafNode() );

        // Find the best split for the node.
        SplitCandidate split = findBestSplit( nodeID );

        // Apply the split if one was found (this will also add the created children to the growable list, if appropriate).
        if ( split.isValid() ) splitNode( nodeID, split );
    }

    /**
     * Returns true iff it is still meaningful to grow the specified node.
     * \pre Node must be a leaf node.
     */
    bool isGrowableNode( NodeID nodeID ) const
    {
        // Find the node and test the precondition.
        auto & node = m_nodes[nodeID];
        assert( node.isLeafNode() );

        // Prohibit growth beyond the maximum depth.
        if ( node.getDistanceToRoot() >= m_maximumDistanceToRoot ) return false;

        // Prohibit the growth of nodes that are already pure enough.
        if ( node.getLabelCounts().template giniImpurity<ImpurityType>() <= m_impurityThreshold ) return false;

        // If there are no further objections, the node is growable.
        return true;
    }

private:

    const Table<FeatureType> &      m_dataPoints;
    const Table<Label> &            m_labels;
    std::deque<NodeID>              m_growableLeaves;
    std::vector<Node>               m_nodes;
    std::vector<SingleFeatureIndex> m_featureIndex;
    WeightedCoinType                m_coin;
    std::size_t                     m_featuresToConsider;
    unsigned int                    m_maximumDistanceToRoot;
    ImpurityType                    m_impurityThreshold;
};

#endif // INDEXEDDECISIONTREE_H
