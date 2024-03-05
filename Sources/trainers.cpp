#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>

#include "decisiontrees.h"
#include "trainers.h"
#include "utilities.h"
#include "weightedcoin.h"

namespace
{
class Mark2TreeTrainer
{
public:

    // typedef DecisionTree<>::NodeID NodeID;
    typedef DecisionTree<>::DecisionTreeNode DecisionTreeNode;

    // Statistics pertaining to each split() call.
    class SplitStatistics
    {
    public:

        SplitStatistics( unsigned int splitsMade = 0, unsigned int pointsLeftToImprove = 0 )
        : splitsMade( splitsMade )
        , pointsLeftToImprove( pointsLeftToImprove )
        {
        }

        unsigned int splitsMade;          // The number of splits made in this pass.
        unsigned int pointsLeftToImprove; // The total number of points that can be classified better after the split.
                                          // N.B. if max. depth is exceeded this will be 0 for those points.
    };

    // Per node annotations used by the training algorithm.
    struct NodeAnnotations
    {
        double lastVisitedValue;
        unsigned int totalCount;          // Total number of points in this node.
        unsigned int trueCount;           // Total number of points labeled 'true' in this node.
        unsigned int totalCountLeftHalf;  // Total number of points that have been visited during traversal of the
                                          // current feature.
        unsigned int trueCountLeftHalf;   // Totol number of visited points labeled 'true'.
        unsigned int trueCountRightHalf;  // Remaining unvisited points labeled 'true'.
        unsigned int currentFeature;      // The feature that is currently being traversed.
        unsigned int bestSplitFeature;    // Best feature for splitting found so far.
        unsigned int bestSplitMislabeled; // Number of mislabeled points that would occur if the split was made here.
        double bestSplitValue;            // Best value to split at found so far.
        double bestSplitGiniIndex;        // Gini-index of the best split point found so far (lowest index).
        unsigned int featuresToConsider;  // The number of randomly chosen features that this node still has to consider
                                          // during the feature traversal phase.
        bool scanningThisFeature; // Whether or not the currently traversed feature is taken into account by this node.
    };

    /**
     * Constructor.
     * \param maxDepth Decision tree depth cut-off used during training.
     */
    Mark2TreeTrainer( unsigned int maxDepth )
    : m_maxDepth( maxDepth )
    {
        m_nodes.reserve( 1024 * 1024 );
        m_annotations.reserve( 1024 * 1024 );
    }

    /**
     * Train a decision tree using the specified feature index and training dataset.
     * \param featureIndex An index for traversing points in a dataset in order of each feature.
     * \param dataSet Set of data points and associated labels to use for training.
     * \return A trained decision tree instance.
     */
    DecisionTree<>::SharedPointer train( const FeatureIndex & featureIndex, const TrainingDataSet & dataSet, unsigned int featuresToScan, bool writeGraphviz, unsigned int treeID )
    {
        // Create a tree containing a single node.
        m_nodes.clear();
        m_nodes.push_back( DecisionTreeNode() );

        // Create a list of per node annotations.
        m_annotations.clear();
        m_annotations.push_back( NodeAnnotations() );

        // Create a mapping from data points to the nodeIDs of their current parent nodes.
        std::vector<NodeID> pointParents( featureIndex.size(), NodeID( 0 ) );

        // Split all leaf nodes in the tree until the there is no more room to improve, or until the depth limit is
        // reached.
        for ( unsigned int depth = 0; depth < m_maxDepth; ++depth )
        {
            std::cout << "Depth = " << depth << std::endl;

            // Determine the number of features to consider during each randomized split. If the supplied value was 0, default to the ceil(sqrt(featurecount)).
            unsigned int numberOfFeatures   = featureIndex.getFeatureCount();
            unsigned int featuresToConsider = featuresToScan ? featuresToScan : std::ceil( std::sqrt( numberOfFeatures ) );
            if ( featuresToConsider > numberOfFeatures ) throw ClientError( "The supplied number of features to scan exceeds the number of features in the dataset." );
            assert( featuresToConsider > 0 );

            // Tell all nodes that a round of optimal split searching is starting.
            initializeOptimalSplitSearch( featuresToConsider );

            // Register all points with their respective parent nodes.
            for ( DataPointID pointID( 0 ), end( pointParents.size() ); pointID < end; ++pointID )
            {
                pointParents[pointID] = registerPoint( pointParents[pointID], dataSet, pointID );
                assert( pointParents[pointID] < m_nodes.size() );
            }

            // Traverse all data points once for each feature, in order, so the tree nodes can find the best possible
            // split for them.
            for ( unsigned int featureID = 0; featureID < numberOfFeatures;
                  ++featureID ) // TODO: random trees should not use all features.
            {
                // Tell the tree that traversal is starting for this feature.
                startFeatureTraversal( featureID, numberOfFeatures - featureID );

                // Traverse all datapoints in order of this feature.
                for ( auto it( featureIndex.featureBegin( featureID ) ), end( featureIndex.featureEnd( featureID ) );
                      it != end;
                      ++it )
                {
                    // Let the parent node of the data point know that it is being traversed.
                    auto & tuple      = *it;
                    auto featureValue = std::get<0>( tuple );
                    auto label        = std::get<1>( tuple );
                    auto pointID      = std::get<2>( tuple );
                    visitPoint( pointParents[pointID], featureValue, label );
                }
            }

            // Allow all leaf nodes to split, if necessary.
            auto splitStats = split();
            std::cout << splitStats.splitsMade << " splits made." << std::endl;

            // Decide whether it is meaningful to continue.
            // TODO: make stop criterium more subtle.
            if ( splitStats.pointsLeftToImprove == 0 )
            {
                std::cout << "No more room for improvement." << std::endl;
                break;
            }
        }

        // Update the annotations for all leaf nodes.
        initializeOptimalSplitSearch( 0 );
        for ( DataPointID pointID( 0 ), end( pointParents.size() ); pointID < end; ++pointID )
        {
            pointParents[pointID] = registerPoint( pointParents[pointID], dataSet, pointID );
            assert( pointParents[pointID] < m_nodes.size() );
        }

        // Decide the label value for all nodes.
        for ( NodeID nodeID( 0 ), end( m_nodes.size() ); nodeID != end; ++nodeID )
        {
            DecisionTreeNode & node = m_nodes[nodeID];
            const NodeAnnotations & nodeStats = m_annotations[nodeID];
            node.label = nodeStats.totalCount < 2 * nodeStats.trueCount;
        }

        // Write a Graphviz file for the tree, if necessary.
        if ( writeGraphviz )
        {
            std::stringstream ss;
            ss << "tree#" << treeID << ".dot";
            this->writeGraphviz( ss.str() );
        }

        // Return a stripped version of the training tree.
        return DecisionTree<>::SharedPointer( new DecisionTree<>( m_nodes.begin(), m_nodes.end() ) );
    }

    /**
     * Initializes the search for the optimal split.
     * \param featuresToConsider The number of randomly selected features that this node will consider during traversal.
     */
    void initializeOptimalSplitSearch( unsigned int featuresToConsider )
    {
        for ( NodeID nodeID( 0 ), end( m_nodes.size() ); nodeID != end; ++nodeID )
        {
            if ( isLeafNode<double, bool>( m_nodes[nodeID] ) )
            {
                // Reset the point counts. Points will be re-counted during the point registration phase.
                auto & nodeStats = m_annotations[nodeID];
                nodeStats.trueCount  = 0;
                nodeStats.totalCount = 0;

                // Reset the number of features that will be considered by this node.
                nodeStats.featuresToConsider = featuresToConsider;

                // Reset the best split found so far. This will be re-determined during the feature traversal phase.
                nodeStats.bestSplitFeature    = 0;
                nodeStats.bestSplitMislabeled = 0;
                nodeStats.bestSplitValue      = 0;
                nodeStats.bestSplitGiniIndex  = std::numeric_limits<double>::max();
            }
        }
    }

    /**
     * Allows this node to count a data point as one of its descendants, and returns the leaf-node that contains the
     * point. \return A pointer to the leaf node that contains the point (direct parent).
     */
    unsigned int registerPoint( NodeID nodeID, const TrainingDataSet & dataSet, DataPointID pointID )
    {
        while ( true )
        {
            const DecisionTreeNode & node = m_nodes[nodeID];
            if ( isLeafNode<double, bool>( node ) ) break;

            // Defer the registration to the correct child.
            // N.B. The comparison must be strictly-less. DO NOT change to <=, or the algorithm will break.
            if ( dataSet.getFeatureValue( pointID, node.splitFeatureID ) < node.splitValue )
                nodeID = node.leftChildID;
            else
                nodeID = node.rightChildID;
        }

        // Count the point if this is a leaf-node, and return this node as the parent.
        NodeAnnotations & nodeStats = m_annotations[nodeID];
        bool label                  = dataSet.getLabel( pointID );
        ++nodeStats.totalCount;
        if ( label ) ++nodeStats.trueCount;
        return nodeID;
    }

    /**
     * Instructs this node and its children that a particular feature will be traversed in-order now.
     * \param featureID The ID of the feature that will be traversed.
     * \param featuresLeft The total number of features that still have to be traversed, including this one.
     */
    void startFeatureTraversal( unsigned int featureID, unsigned int featuresLeft )
    {
        for ( NodeID nodeID( 0 ), end( m_nodes.size() ); nodeID != end; ++nodeID )
        {
            if ( !isLeafNode<double, bool>( m_nodes[nodeID] ) ) continue;

            // Reset the feature traversal statistics.
            NodeAnnotations & nodeStats  = m_annotations[nodeID];
            nodeStats.lastVisitedValue   = std::numeric_limits<double>::min(); // Arbitrary.
            nodeStats.totalCountLeftHalf = 0;
            nodeStats.trueCountLeftHalf  = 0;
            nodeStats.trueCountRightHalf = nodeStats.trueCount;
            nodeStats.currentFeature     = featureID;

            // Determine whether or not this node will consider this feature during this pass.
            nodeStats.scanningThisFeature = m_coin.flip( nodeStats.featuresToConsider, featuresLeft );
            if ( nodeStats.scanningThisFeature )
            {
                assert( nodeStats.featuresToConsider > 0 );
                --nodeStats.featuresToConsider; // Use up one 'credit'.
            }
        }
    }

    /**
     * Visit one point during the feature traversal phase.
     */
    void visitPoint( NodeID nodeID, double featureValue, bool label )
    {
        // Do nothing if this node is not considering this feature.
        NodeAnnotations & nodeStats = m_annotations[nodeID];
        if ( !nodeStats.scanningThisFeature ) return;

        // If this is the start of a block of previously unseen feature values, calculate what the gain of a split would
        // be.
        if ( ( featureValue != nodeStats.lastVisitedValue ) && ( nodeStats.totalCountLeftHalf > 0 ) )
        {
            // Compute the Gini gain, assuming a split is made at this point.
            auto totalCountRightHalf = nodeStats.totalCount - nodeStats.totalCountLeftHalf;
            auto giniLeft            = giniImpurity( nodeStats.trueCountLeftHalf, nodeStats.totalCountLeftHalf );
            auto giniRight           = giniImpurity( nodeStats.trueCountRightHalf, totalCountRightHalf );
            auto giniTotal =
                ( giniLeft * nodeStats.totalCountLeftHalf + giniRight * totalCountRightHalf ) / nodeStats.totalCount;

            // Save this split if it is the best one so far.
            if ( giniTotal < nodeStats.bestSplitGiniIndex )
            {
                auto falseCountLeft           = nodeStats.totalCountLeftHalf - nodeStats.trueCountLeftHalf;
                auto falseCountRight          = totalCountRightHalf - nodeStats.trueCountRightHalf;
                nodeStats.bestSplitFeature    = nodeStats.currentFeature;
                nodeStats.bestSplitMislabeled = std::min( nodeStats.trueCountLeftHalf, falseCountLeft ) +
                                                std::min( nodeStats.trueCountRightHalf, falseCountRight );
                nodeStats.bestSplitValue     = featureValue;
                nodeStats.bestSplitGiniIndex = giniTotal;
            }
        }

        // Count this point and its label. The point now belongs to the 'left half' of the pass for this feature.
        if ( label )
        {
            ++nodeStats.trueCountLeftHalf;
            --nodeStats.trueCountRightHalf;
        }
        ++nodeStats.totalCountLeftHalf;

        // Update the last visited value. This is necessary to detect the end of a block during the visit of the next
        // point.
        nodeStats.lastVisitedValue = featureValue;
    }

    /**
     * Split the leaf nodes at the most optimal point, after all features have been traversed.
     * \param maxDepth Provide the maximum allowed depth.
     * \param depth Keep to default value of 0.
     * \return Statistics pertaining to the split.
     */
    SplitStatistics split()
    {
        SplitStatistics splitStats;
        for ( NodeID nodeID = NodeID( 0 ), end( m_nodes.size() ); nodeID != end; ++nodeID )
        {
            // Do not attempt to split interior nodes.
            if ( !isLeafNode<double, bool>( m_nodes[nodeID] ) ) continue;

            // Determine whether it's time to stop permanently.
            const NodeAnnotations & nodeStats = m_annotations[nodeID];
            if ( nodeStats.trueCount == 0 || nodeStats.trueCount == nodeStats.totalCount ) continue;

            // Split this node at the best point that was found. First, add two
            // new nodes to the tree. Note that this could cause the vector of
            // nodes to be resized, which would invalidate any reference to
            // tree nodes. Take care when moving this code.
            m_nodes.push_back( DecisionTreeNode() );
            const NodeID leftChildID = m_nodes.size() - 1;
            m_nodes.push_back( DecisionTreeNode() );
            const NodeID rightChildID = m_nodes.size() - 1;

            // Update node attributes.
            DecisionTreeNode & node = m_nodes[nodeID];
            node.leftChildID        = leftChildID;
            node.rightChildID       = rightChildID;
            node.splitFeatureID     = nodeStats.bestSplitFeature;
            node.splitValue         = nodeStats.bestSplitValue;

            // Update statistics.
            splitStats.splitsMade++;
            splitStats.pointsLeftToImprove += nodeStats.bestSplitMislabeled;

            // Add training statistics for the newly added nodes. Note that this
            // may cause the vector of training statistics to be resized, which
            // would invalidate any reference to node statistics. Take care
            // when moving this code.
            m_annotations.push_back( NodeAnnotations() );
            m_annotations.push_back( NodeAnnotations() );
        }
        return splitStats;
    }

    /**
     * Write the tree model to a Dotty file, suitable for visualization.
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
            auto &stats = m_annotations[nodeID];
            std::stringstream info;
            info << 'N' << nodeID << " = " << static_cast<int>( node.label ) << " counts: " << (stats.totalCount - stats.trueCount) << " " << stats.trueCount;
            out << "    node" << nodeID << "[shape=box label=\"" << info.str() << "\"]" << std::endl;

            // Write the links to the children.
            if ( !isLeafNode<double, bool>( node ) )
            {
                auto splitFeature = node.splitFeatureID;
                auto splitValue   = node.splitValue;
                out << "    node" << nodeID << " -> " << "node" << node.leftChildID  << " [label=\"F" << static_cast<int>( splitFeature ) << " < " << splitValue << "\"];" << std::endl;
                out << "    node" << nodeID << " -> " << "node" << node.rightChildID << ';' << std::endl;
            }
        }
        out << "}" << std::endl;

        // Close the file.
        out.close();
    }

    void dump( NodeID nodeID = 0, unsigned int indent = 0 )
    {
        const DecisionTreeNode & node     = m_nodes[nodeID];
        const NodeAnnotations & nodeStats = m_annotations[nodeID];

        auto tab = std::string( indent, ' ' );
        std::cout << tab << "Node:" << std::endl
                  << tab << "m_totalCount          = " << nodeStats.totalCount << std::endl
                  << tab << "m_trueCount           = " << nodeStats.trueCount << std::endl
                  << tab << "m_totalCountLeftHalf  = " << nodeStats.totalCountLeftHalf << std::endl
                  << tab << "m_trueCountLeftHalf   = " << nodeStats.trueCountLeftHalf << std::endl
                  << tab << "m_trueCountRightHalf  = " << nodeStats.trueCountRightHalf << std::endl
                  << tab << "m_currentFeature      = " << nodeStats.currentFeature << std::endl
                  << tab << "m_bestSplitFeature    = " << nodeStats.bestSplitFeature << std::endl
                  << tab << "m_bestSplitMislabeled = " << nodeStats.bestSplitMislabeled << std::endl
                  << tab << "m_bestSplitValue      = " << nodeStats.bestSplitValue << std::endl
                  << tab << "m_bestSplitGiniIndex  = " << nodeStats.bestSplitGiniIndex << std::endl;

        if ( !isLeafNode<double, bool>( node ) )
        {
            std::cout << tab << "Split feature #" << node.splitFeatureID << ", value = " << node.splitValue
                      << std::endl;
            std::cout << tab << "Left: " << std::endl;
            dump( node.leftChildID, indent + 1 );
            std::cout << tab << "Right: " << std::endl;
            dump( node.rightChildID, indent + 1 );
        }
    }

    unsigned int m_maxDepth;
    WeightedCoin m_coin;
    std::vector<NodeAnnotations> m_annotations;
    std::vector<DecisionTreeNode> m_nodes;
};
} // Anonymous namespace.

template <>
DecisionTree<>::SharedPointer SingleTreeTrainerMark2<>::train( const FeatureIndex & featureIndex,
    const TrainingDataSet & dataSet, unsigned int featuresToScan, bool writeGraphviz, unsigned int treeID )
{
    Mark2TreeTrainer trainer( m_maxDepth );
    return trainer.train( featureIndex, dataSet, featuresToScan, writeGraphviz, treeID );
}
