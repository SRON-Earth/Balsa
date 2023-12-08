#include <algorithm>
#include <cassert>
#include <cmath>

#include "decisiontrees.h"
#include "trainermark2.h"
#include "utilities.h"

namespace
{
// Statistics pertaining to each split() call.
class SplitStatistics
{
public:

  SplitStatistics( unsigned int splitsMade = 0, unsigned int pointsLeftToImprove = 0 ):
  splitsMade         ( splitsMade          ),
  pointsLeftToImprove( pointsLeftToImprove )
  {
  }

  SplitStatistics( const SplitStatistics &left, const SplitStatistics &right ):
  splitsMade( left.splitsMade + right.splitsMade ),
  pointsLeftToImprove( left.pointsLeftToImprove + right.pointsLeftToImprove )
  {
  }

  unsigned int splitsMade         ; // The number of splits made in this pass.
  unsigned int pointsLeftToImprove; // The total number of points that can be classified better after the split. N.B. if max. depth is exceeded this will be 0 for those points.
};

struct NodeTrainingStatistics
{
    double       lastVisitedValue   ;
    unsigned int totalCount         ; // Total number of points in this node.
    unsigned int trueCount          ; // Total number of points labeled 'true' in this node.
    unsigned int totalCountLeftHalf ; // Total number of points that have been visited during traversal of the current feature.
    unsigned int trueCountLeftHalf  ; // Totol number of visited points labeled 'true'.
    unsigned int trueCountRightHalf ; // Remaining unvisited points labeled 'true'.
    unsigned int currentFeature     ; // The feature that is currently being traversed.
    unsigned int bestSplitFeature   ; // Best feature for splitting found so far.
    unsigned int bestSplitMislabeled; // Number of mislabeled points that would occur if the split was made here.
    double       bestSplitValue     ; // Best value to split at found so far.
    double       bestSplitGiniIndex ; // Gini-index of the best split point found so far (lowest index).
    unsigned int featuresToConsider ; // The number of randomly chosen features that this node still has to consider during the feature traversal phase.
    bool         ignoringThisFeature; // Whether or not the currently traversed feature is taken into account by this node.
};
}

typedef unsigned int NodeID;
typedef std::vector<NodeTrainingStatistics> TrainingStatistics;

/**
 * Initializes the search for the optimal split.
 * \param featuresToConsider The number of randomly selected features that this node will consider during traversal.
 */
void initializeOptimalSplitSearch( TrainingStatistics & trainingStatistics, unsigned int featuresToConsider )
{
    for ( auto & nodeStats : trainingStatistics )
    {
        // Reset the point counts. Points will be re-counted during the point registration phase.
        nodeStats.trueCount           = 0;
        nodeStats.totalCount          = 0;

        // Reset the number of features that will be considered by this node.
        nodeStats.featuresToConsider  = featuresToConsider;

        // Reset the best split found so far. This will be re-determined during the feature traversal phase.
        nodeStats.bestSplitFeature    = 0;
        nodeStats.bestSplitMislabeled = 0;
        nodeStats.bestSplitValue      = 0;
        nodeStats.bestSplitGiniIndex  = std::numeric_limits<double>::max();
    }
}

/**
 * Allows this node to count a data point as one of its descendants, and returns the leaf-node that contains the point.
 * \return A pointer to the leaf node that contains the point (direct parent).
 */
unsigned int registerPoint( DecisionTree & tree, TrainingStatistics & trainingStatistics, const TrainingDataSet &dataset, DataPointID pointID, NodeID nodeID = NodeID() )
{
    const DecisionTreeNode & node = tree.getNode( nodeID );

    // Count the point if this is a leaf-node, and return this node as the parent.
    if ( isLeafNode( node ) )
    {
        NodeTrainingStatistics & nodeStats = trainingStatistics[nodeID];
        bool label = dataset.getLabel( pointID );
        ++nodeStats.totalCount;
        if ( label ) ++nodeStats.trueCount;
        return nodeID;
    }

    // Defer the registration to the correct child.
    // N.B. The comparison must be strictly-less. DO NOT change to <=, or the algorithm will break.
    if ( dataset.getFeatureValue( pointID, node.splitFeatureID ) < node.splitValue )
        return registerPoint( tree, trainingStatistics, dataset, pointID, node.leftChildID );
    return registerPoint( tree, trainingStatistics, dataset, pointID, node.rightChildID );
}

/**
 * Instructs this node and its children that a particular feature will be traversed in-order now.
 * \param featureID The ID of the feature that will be traversed.
 * \param featuresLeft The total number of features that still have to be traversed, including this one.
 */
void startFeatureTraversal( const DecisionTree & tree, TrainingStatistics & trainingStatistics, unsigned int featureID, unsigned int featuresLeft, NodeID nodeID = NodeID() )
{
    // Start feature traversal in the children, if present.
    const DecisionTreeNode & node = tree.getNode( nodeID );
    if ( !isLeafNode( node ) )
    {
        assert( node.rightChildID );
        startFeatureTraversal( tree, trainingStatistics, featureID, featuresLeft, node.leftChildID  );
        startFeatureTraversal( tree, trainingStatistics, featureID, featuresLeft, node.rightChildID );
    }
    else
    {
        // Reset the feature traversal statistics.
        NodeTrainingStatistics & nodeStats = trainingStatistics[nodeID];
        nodeStats.lastVisitedValue   = std::numeric_limits<double>::min(); // Arbitrary.
        nodeStats.totalCountLeftHalf = 0          ;
        nodeStats.trueCountLeftHalf  = 0          ;
        nodeStats.trueCountRightHalf = nodeStats.trueCount;
        nodeStats.currentFeature     = featureID  ;

        // Determine whether or not this node will consider this feature during this pass.
        nodeStats.ignoringThisFeature = randomBool( nodeStats.featuresToConsider, featuresLeft );
        if ( nodeStats.ignoringThisFeature )
        {
            assert( nodeStats.featuresToConsider > 0 );
            --nodeStats.featuresToConsider; // Use up one 'credit'.
        }
    }
}

/**
 * Visit one point during the feature traversal phase.
 */
void visitPoint( NodeTrainingStatistics &nodeStats, double featureValue, bool label )
{
    // Do nothing if this node is not considering this feature.
    if ( nodeStats.ignoringThisFeature ) return;

    // If this is the start of a block of previously unseen feature values, calculate what the gain of a split would be.
    if ( ( featureValue != nodeStats.lastVisitedValue ) && ( nodeStats.totalCountLeftHalf > 0 ) )
    {
        // Compute the Gini gain, assuming a split is made at this point.
        auto totalCountRightHalf = nodeStats.totalCount - nodeStats.totalCountLeftHalf;
        auto giniLeft  = giniImpurity( nodeStats.trueCountLeftHalf , nodeStats.totalCountLeftHalf );
        auto giniRight = giniImpurity( nodeStats.trueCountRightHalf, totalCountRightHalf  );
        auto giniTotal = ( giniLeft * nodeStats.totalCountLeftHalf + giniRight * totalCountRightHalf ) / nodeStats.totalCount;

        // Save this split if it is the best one so far.
        if ( giniTotal < nodeStats.bestSplitGiniIndex )
        {
            auto falseCountLeft   = nodeStats.totalCountLeftHalf  - nodeStats.trueCountLeftHalf ;
            auto falseCountRight  = totalCountRightHalf - nodeStats.trueCountRightHalf;
            nodeStats.bestSplitFeature    = nodeStats.currentFeature;
            nodeStats.bestSplitMislabeled = std::min( nodeStats.trueCountLeftHalf, falseCountLeft ) + std::min( nodeStats.trueCountRightHalf, falseCountRight );
            nodeStats.bestSplitValue      = featureValue;
            nodeStats.bestSplitGiniIndex  = giniTotal   ;
        }
    }

    // Count this point and its label. The point now belongs to the 'left half' of the pass for this feature.
    if ( label )
    {
        ++nodeStats.trueCountLeftHalf;
        --nodeStats.trueCountRightHalf;
    }
    ++nodeStats.totalCountLeftHalf;

    // Update the last visited value. This is necessary to detect the end of a block during the visit of the next point.
    nodeStats.lastVisitedValue = featureValue;
}

/**
 * Split the leaf nodes at the most optimal point, after all features have been traversed.
 * \param maxDepth Provide the maximum allowed depth.
 * \param depth Keep to default value of 0.
 * \return Statistics pertaining to the split.
 */
SplitStatistics split( DecisionTree & tree, TrainingStatistics & trainingStatistics, unsigned int maxDepth, unsigned int depth = 0, NodeID nodeID = NodeID() )
{
    // If this is an interior node, split the children and quit.
    DecisionTreeNode & node = tree.getNode( nodeID );
    // std::cout << ">>> split node #" << nodeID << " leaf=" << isLeafNode(node) << std::endl;
    if ( !isLeafNode( node ) )
    {
        assert( node.rightChildID );
        const NodeID leftChildID = node.leftChildID;
        const NodeID rightChildID = node.rightChildID;
        // std::cout << "### split node #" << nodeID << " left=" << leftChildID << " right=" << rightChildID << std::endl;
        auto l = split( tree, trainingStatistics, maxDepth, depth + 1, leftChildID  );
        auto r = split( tree, trainingStatistics, maxDepth, depth + 1, rightChildID );
        return SplitStatistics( l, r );
    }

    NodeTrainingStatistics & nodeStats = trainingStatistics[nodeID];

    // Determine whether it's time to stop permanently.
    if ( nodeStats.trueCount == 0 || nodeStats.trueCount == nodeStats.totalCount )
    {
        return SplitStatistics();
    }

    // std::cout << "==================" << std::endl;
    // std::cout << "split node #" << nodeID << " feature=" << nodeStats.bestSplitFeature << " value=" << nodeStats.bestSplitValue << std::endl;
    // std::cout << "==================" << std::endl;

    // Split this node at the best point that was found.
    const NodeID leftChildID  = tree.addNode( DecisionTreeNode() );
    const NodeID rightChildID = tree.addNode( DecisionTreeNode() );

    DecisionTreeNode & node2 = tree.getNode( nodeID );
    node2.leftChildID    = leftChildID;
    node2.rightChildID   = rightChildID;
    node2.splitFeatureID = nodeStats.bestSplitFeature;
    node2.splitValue     = nodeStats.bestSplitValue;

    SplitStatistics splitStats( 1, nodeStats.bestSplitMislabeled );

    trainingStatistics.push_back( NodeTrainingStatistics() );
    trainingStatistics.push_back( NodeTrainingStatistics() );

    assert( trainingStatistics.size() == tree.getNodeCount() );

    return splitStats;
}

void finalize( DecisionTree & tree, const TrainingStatistics & trainingStatistics )
{
    for ( NodeID nodeID = 0, end = tree.getNodeCount(); nodeID != end; ++nodeID )
    {
        DecisionTreeNode & node = tree.getNode( nodeID );
        if ( isLeafNode( node ) )
        {
            const NodeTrainingStatistics & nodeStats = trainingStatistics[nodeID];
            node.label = nodeStats.totalCount < 2 * nodeStats.trueCount;
        }
    }
}

// Debug
void dump( const DecisionTree & tree, const TrainingStatistics & trainingStatistics, unsigned int indent = 0, NodeID nodeID = NodeID() )
{
    const DecisionTreeNode & node = tree.getNode( nodeID );
    const NodeTrainingStatistics & nodeStats = trainingStatistics[nodeID];

    auto tab = std::string( indent, ' ' );
    std::cout << tab << "Node:" << std::endl // " (" << nodeID << "):" << std::endl
              << tab << "m_totalCount          = " << nodeStats.totalCount          << std::endl
              << tab << "m_trueCount           = " << nodeStats.trueCount           << std::endl
              << tab << "m_totalCountLeftHalf  = " << nodeStats.totalCountLeftHalf  << std::endl
              << tab << "m_trueCountLeftHalf   = " << nodeStats.trueCountLeftHalf   << std::endl
              << tab << "m_trueCountRightHalf  = " << nodeStats.trueCountRightHalf  << std::endl
              << tab << "m_currentFeature      = " << nodeStats.currentFeature      << std::endl
              << tab << "m_bestSplitFeature    = " << nodeStats.bestSplitFeature    << std::endl
              << tab << "m_bestSplitMislabeled = " << nodeStats.bestSplitMislabeled << std::endl
              << tab << "m_bestSplitValue      = " << nodeStats.bestSplitValue      << std::endl
              << tab << "m_bestSplitGiniIndex  = " << nodeStats.bestSplitGiniIndex  << std::endl;

    if ( !isLeafNode( node ) )
    {
        std::cout << tab << "Split feature #" << node.splitFeatureID << ", value = " <<  node.splitValue << std::endl;
        std::cout << tab << "Left: " << std::endl;
        dump( tree, trainingStatistics, indent + 1, node.leftChildID );
        std::cout << tab << "Right: " << std::endl;
        dump( tree, trainingStatistics, indent + 1, node.rightChildID );
    }
}

DecisionTree::SharedPointer SingleTreeTrainerMark2::train( const FeatureIndex &featureIndex, const TrainingDataSet &dataSet )
{
    // Create an empty training tree.
    DecisionTree::SharedPointer tree( new DecisionTree( 1000 ) );

    // Create a list of pointers from data points to their current parent nodes.
    std::vector<NodeID> pointParents( featureIndex.size(), NodeID() );

    TrainingStatistics trainingStatistics;
    trainingStatistics.reserve( 1000 );

    // Create root node.
    tree->addNode( DecisionTreeNode() );
    trainingStatistics.push_back( NodeTrainingStatistics() );

    // Split all leaf nodes in the tree until the there is no more room to improve, or until the depth limit is reached.
    for ( unsigned int depth = 1; depth < m_maxDepth; ++depth )
    {
        std::cout << "Depth = " << depth << std::endl;\

        // Tell all nodes that a round of optimal split searching is starting.
        unsigned int featureCount = featureIndex.getFeatureCount();
        unsigned int featuresToConsider = std::ceil( std::sqrt( featureCount ) );
        assert( featuresToConsider > 0 );
        initializeOptimalSplitSearch( trainingStatistics, featuresToConsider );

        // Register all points with their respective parent nodes.
        for ( DataPointID pointID( 0 ), end( pointParents.size() ); pointID < end; ++pointID )
        {
            pointParents[pointID] = registerPoint( *tree, trainingStatistics, dataSet, pointID, pointParents[pointID] );
            assert( pointParents[pointID] < tree->getNodeCount() );
        }

        // Traverse all data points once for each feature, in order, so the tree nodes can find the best possible split for them.
        for ( unsigned int featureID = 0; featureID < featureCount; ++featureID ) // TODO: random trees should not use all features.
        {
            // Tell the tree that traversal is starting for this feature.
            startFeatureTraversal( *tree, trainingStatistics, featureID, featureCount - featureID );

            // Traverse all datapoints in order of this feature.
            for ( auto it( featureIndex.featureBegin( featureID ) ), end( featureIndex.featureEnd( featureID ) ); it != end; ++it )
            {
                // Let the parent node of the data point know that it is being traversed.
                auto &tuple = *it;
                auto featureValue = std::get<0>( tuple );
                auto label        = std::get<1>( tuple );
                auto pointID      = std::get<2>( tuple );
                NodeTrainingStatistics & nodeStats = trainingStatistics[pointParents[pointID]];
                visitPoint( nodeStats, featureValue, label );
            }
        }

        // Allow all leaf nodes to split, if necessary.
        auto splitStats = split( *tree, trainingStatistics, m_maxDepth );
        std::cout << splitStats.splitsMade << " splits made." << std::endl;

        // Decide whether it is meaningful to continue.
        // TODO: make stop criterium more subtle.
        if ( splitStats.pointsLeftToImprove == 0 )
        {
            std::cout << "No more room for improvement." << std::endl;
            break;
        }
    }

    finalize( *tree, trainingStatistics );

    // Return a stripped version of the training tree.
    return tree;
}
