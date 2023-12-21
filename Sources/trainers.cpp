#include "trainers.h"
#include "utilities.h"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace
{
  /**
   * A node in a decision tree, with special annotations for the training process.
   */
  class Mark1TrainingTreeNode
  {
  public:

    typedef std::shared_ptr<Mark1TrainingTreeNode> SharedPointer;

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

    Mark1TrainingTreeNode( Mark1TrainingTreeNode *parent = nullptr ):
    m_parent             ( parent ),
    m_splitFeatureID     ( 0      ),
    m_splitValue         ( 0      ),
    m_totalCount         ( 0      ),
    m_trueCount          ( 0      ),
    m_totalCountLeftHalf ( 0      ),
    m_trueCountLeftHalf  ( 0      ),
    m_trueCountRightHalf ( 0      ),
    m_currentFeature     ( 0      ),
    m_bestSplitFeature   ( 0      ),
    m_bestSplitMislabeled( 0      ),
    m_bestSplitValue     ( 0      ),
    m_bestSplitGiniIndex ( 0      ),
    m_featuresToConsider ( 0      ),
    m_ignoringThisFeature( false  )
    {
    }

    /**
     * Allows this node to count a data point as one of its descendants, and returns the leaf-node that contains the point.
     * \return A pointer to the leaf node that contains the point (direct parent).
     */
    Mark1TrainingTreeNode *registerPoint( DataPointID pointID, const TrainingDataSet &dataset )
    {
        // Count the point if this is a leaf-node, and return this node as the parent.
        if ( !m_leftChild )
        {
            bool label = dataset.getLabel( pointID );
            assert( !m_rightChild );
            ++m_totalCount;
            if ( label ) ++m_trueCount;
            return this;
        }

        // Defer the registration to the correct child.
        // N.B. The comparison must be strictly-less. DO NOT change to <=, or the algorithm will break.
        if ( dataset.getFeatureValue( pointID, m_splitFeatureID ) < m_splitValue ) return m_leftChild->registerPoint( pointID, dataset );
        return m_rightChild->registerPoint( pointID, dataset );
    }

    // Debug
    void dump( unsigned int indent = 0 ) const
    {

        auto tab = std::string( indent, ' ' );
        std::cout << tab << "Node:" << std::endl
                  << tab << "m_totalCount          = " << m_totalCount          << std::endl
                  << tab << "m_trueCount           = " << m_trueCount           << std::endl
                  << tab << "m_totalCountLeftHalf  = " << m_totalCountLeftHalf  << std::endl
                  << tab << "m_trueCountLeftHalf   = " << m_trueCountLeftHalf   << std::endl
                  << tab << "m_trueCountRightHalf  = " << m_trueCountRightHalf  << std::endl
                  << tab << "m_currentFeature      = " << m_currentFeature      << std::endl
                  << tab << "m_bestSplitFeature    = " << m_bestSplitFeature    << std::endl
                  << tab << "m_bestSplitMislabeled = " << m_bestSplitMislabeled << std::endl
                  << tab << "m_bestSplitValue      = " << m_bestSplitValue      << std::endl
                  << tab << "m_bestSplitGiniIndex  = " << m_bestSplitGiniIndex  << std::endl;

        if ( m_leftChild )
        {
            std::cout << tab << "Split feature #" << m_splitFeatureID << ", value = " <<  m_splitValue << std::endl;
            std::cout << tab << "Left: " << std::endl;
            m_leftChild->dump( indent + 1 );
            std::cout << tab << "Right: " << std::endl;
            m_rightChild->dump( indent + 1 );
        }
    }


    /**
     * Returns a stripped, non-training decision tree.
     */
    DecisionTree::SharedPointer finalize()
    {
        DecisionTree::SharedPointer tree( new DecisionTree() );
        unsigned int targetDepth = 0;
        while ( finalize( *tree, targetDepth ) )
            ++targetDepth;
        return tree;
    }

    /**
     * Returns the most obvious label for this node.
     */
    bool getLabel() const
    {
        return m_totalCount < 2 * m_trueCount;
    }

    /**
     * Initializes the search for the optimal split.
     * \param featuresToConsider The number of randomly selected features that this node will consider during traversal.
     */
    void initializeOptimalSplitSearch( unsigned int featuresToConsider )
    {
        // Reset the point counts. Points will be re-counted during the point registration phase.
        m_trueCount          = 0;
        m_totalCount         = 0;

        // Reset the number of features that will be considered by this node.
        m_featuresToConsider = featuresToConsider;

        // Reset the best split found so far. This will be re-determined during the feature traversal phase.
        m_bestSplitFeature    = 0;
        m_bestSplitMislabeled = 0;
        m_bestSplitValue      = 0;
        m_bestSplitGiniIndex  = std::numeric_limits<double>::max();

        // Initialize any children.
        if ( m_leftChild  ) m_leftChild ->initializeOptimalSplitSearch( featuresToConsider );
        if ( m_rightChild ) m_rightChild->initializeOptimalSplitSearch( featuresToConsider );
    }

    /**
     * Instructs this node and its children that a particular feature will be traversed in-order now.
     * \param featureID The ID of the feature that will be traversed.
     * \param featuresLeft The total number of features that still have to be traversed, including this one.
     */
    void startFeatureTraversal( unsigned int featureID, unsigned int featuresLeft, WeightedCoin & coin )
    {
        // Start feature traversal in the children, if present.
        if ( m_leftChild )
        {
            assert( m_rightChild );
            m_leftChild->startFeatureTraversal ( featureID, featuresLeft, coin );
            m_rightChild->startFeatureTraversal( featureID, featuresLeft, coin );
        }
        else
        {
            // Reset the feature traversal statistics.
            m_lastVisitedValue    = std::numeric_limits<double>::min(); // Arbitrary.
            m_totalCountLeftHalf  = 0          ;
            m_trueCountLeftHalf   = 0          ;
            m_trueCountRightHalf  = m_trueCount;
            m_currentFeature      = featureID  ;

            // Determine whether or not this node will consider this feature during this pass.
            m_ignoringThisFeature = coin.flip( m_featuresToConsider, featuresLeft );
            if ( m_ignoringThisFeature )
            {
                assert( m_featuresToConsider > 0 );
                --m_featuresToConsider; // Use up one 'credit'.
            }
        }
    }

    /**
     * Visit one point during the feature traversal phase.
     */
    void visitPoint( DataPointID, double featureValue, bool label )
    {
        // This must never be called on internal nodes.
        assert( !m_leftChild  );
        assert( !m_rightChild );

        // Do nothing if this node is not considering this feature.
        if ( m_ignoringThisFeature ) return;

        // If this is the start of a block of previously unseen feature values, calculate what the gain of a split would be.
        if ( ( featureValue != m_lastVisitedValue ) && ( m_totalCountLeftHalf > 0 ) )
        {
            // Compute the Gini gain, assuming a split is made at this point.
            auto totalCountRightHalf = m_totalCount - m_totalCountLeftHalf;
            auto giniLeft  = giniImpurity( m_trueCountLeftHalf , m_totalCountLeftHalf );
            auto giniRight = giniImpurity( m_trueCountRightHalf, totalCountRightHalf  );
            auto giniTotal = ( giniLeft * m_totalCountLeftHalf + giniRight * totalCountRightHalf ) / m_totalCount;

            // Save this split if it is the best one so far.
            if ( giniTotal < m_bestSplitGiniIndex )
            {
                auto falseCountLeft   = m_totalCountLeftHalf  - m_trueCountLeftHalf ;
                auto falseCountRight  = totalCountRightHalf - m_trueCountRightHalf;
                m_bestSplitFeature    = m_currentFeature;
                m_bestSplitMislabeled = std::min( m_trueCountLeftHalf, falseCountLeft ) + std::min( m_trueCountRightHalf, falseCountRight );
                m_bestSplitValue      = featureValue    ;
                m_bestSplitGiniIndex  = giniTotal       ;
            }
        }

        // Count this point and its label. The point now belongs to the 'left half' of the pass for this feature.
        if ( label )
        {
            ++m_trueCountLeftHalf;
            --m_trueCountRightHalf;
        }
        ++m_totalCountLeftHalf;

        // Update the last visited value. This is necessary to detect the end of a block during the visit of the next point.
        m_lastVisitedValue = featureValue;
    }

    /**
     * Split the leaf nodes at the most optimal point, after all features have been traversed.
     * \param maxDepth Provide the maximum allowed depth.
     * \param depth Keep to default value of 0.
     * \return Statistics pertaining to the split.
     */
    SplitStatistics split( unsigned int maxDepth, unsigned int depth = 0 )
    {
        // If this is an interior node, split the children and quit.
        if ( m_leftChild )
        {
            assert( m_rightChild );
            auto l = m_leftChild ->split( maxDepth, depth + 1 );
            auto r = m_rightChild->split( maxDepth, depth + 1 );
            SplitStatistics stats( l, r );
            return stats;
        }

        // Assert that this is a leaf node.
        assert( !m_leftChild  );
        assert( !m_rightChild );

        // Determine whether it's time to stop permanently.
        if ( m_trueCount == 0 || m_trueCount == m_totalCount )
        {
            return SplitStatistics();
        }

        // Split this node at the best point that was found.
        m_splitValue     = m_bestSplitValue;
        m_splitFeatureID = m_bestSplitFeature;
        m_leftChild .reset( new Mark1TrainingTreeNode( this ) );
        m_rightChild.reset( new Mark1TrainingTreeNode( this ) );

        return SplitStatistics( 1, m_bestSplitMislabeled );
    }

    /**
     * Returns the path of this node from the root (debug purposes).
     */
    std::string getPath() const
    {
        if ( !m_parent ) return "root";
        return m_parent->getPath() + ( m_parent->isLeftChild( this ) ? ".L" : ".R" );
    }

  private:

    bool finalize( DecisionTree & tree, unsigned int targetDepth, unsigned int depth = 0, unsigned int parentNodeID = 0 )
    {
        const bool isInternalNode = static_cast<bool>( m_leftChild );
        assert( !isInternalNode || m_rightChild );

        if ( depth == targetDepth )
        {
            // Add node.
            const unsigned int nodeID = tree.addNode( DecisionTree::DecisionTreeNode() );

            // Copy node attributes.
            DecisionTree::DecisionTreeNode & node = tree[nodeID];
            if ( isInternalNode )
            {
                node.splitFeatureID = m_splitFeatureID;
                node.splitValue = m_splitValue;
            }
            else
            {
                node.label = getLabel();
            }

            // Register child node with its parent.
            if ( m_parent )
            {
                DecisionTree::DecisionTreeNode & parentNode = tree[parentNodeID];
                if ( m_parent->isLeftChild( this ) )
                    parentNode.leftChildID = nodeID;
                else
                    parentNode.rightChildID = nodeID;
            }

            return isInternalNode;
        }

        if ( isInternalNode )
        {
            assert( parentNodeID == 0 || depth > 0 );

            if ( m_parent )
            {
                DecisionTree::DecisionTreeNode & parentNode = tree[parentNodeID];
                parentNodeID = m_parent->isLeftChild( this ) ? parentNode.leftChildID : parentNode.rightChildID;
            }

            const bool leftContinue  = m_leftChild ->finalize( tree, targetDepth, depth + 1, parentNodeID );
            const bool rightContinue = m_rightChild->finalize( tree, targetDepth, depth + 1, parentNodeID );

            return leftContinue || rightContinue;
        }

        // Leaf node.
        return false;
    }

    bool isLeftChild( const Mark1TrainingTreeNode *node ) const
    {
        return m_leftChild.get() == node;
    }

    Mark1TrainingTreeNode *m_parent            ;
    SharedPointer     m_leftChild         ; // Null for leaf nodes.
    SharedPointer     m_rightChild        ; // Null for leaf nodes.
    unsigned int      m_splitFeatureID    ; // The ID of the feature at which this node is split. Only valid for internal nodes.
    double            m_splitValue        ; // The value at which this node is split, along the specified feature.
    unsigned int      m_totalCount        ; // Total number of points in this node.
    unsigned int      m_trueCount         ; // Total number of points labeled 'true' in this node.

    // Statistics used during traversal:
    double            m_lastVisitedValue   ;
    unsigned int      m_totalCountLeftHalf ; // Total number of points that have been visited during traversal of the current feature.
    unsigned int      m_trueCountLeftHalf  ; // Totol number of visited points labeled 'true'.
    unsigned int      m_trueCountRightHalf ; // Remaining unvisited points labeled 'true'.
    unsigned int      m_currentFeature     ; // The feature that is currently being traversed.
    unsigned int      m_bestSplitFeature   ; // Best feature for splitting found so far.
    unsigned int      m_bestSplitMislabeled; // Number of mislabeled points that would occur if the split was made here.
    double            m_bestSplitValue     ; // Best value to split at found so far.
    double            m_bestSplitGiniIndex ; // Gini-index of the best split point found so far (lowest index).
    unsigned int      m_featuresToConsider ; // The number of randomly chosen features that this node still has to consider during the feature traversal phase.
    bool              m_ignoringThisFeature; // Whether or not the currently traversed feature is taken into account by this node.

  };
}

DecisionTree::SharedPointer SingleTreeTrainerMark1::train( const FeatureIndex &featureIndex, const TrainingDataSet &dataSet )
{
    // Create an empty training tree.
    Mark1TrainingTreeNode root;

    // Create a list of pointers from data points to their current parent nodes.
    std::vector< Mark1TrainingTreeNode * > pointParents( featureIndex.size(), &root );

    // Split all leaf nodes in the tree until the there is no more room to improve, or until the depth limit is reached.
    for ( unsigned int depth = 1; depth < m_maxDepth; ++depth )
    {
        std::cout << "Depth = " << depth << std::endl;

        // Tell all nodes that a round of optimal split searching is starting.
        unsigned int featureCount = featureIndex.getFeatureCount();
        unsigned int featuresToConsider = std::ceil( std::sqrt( featureCount ) );
        assert( featuresToConsider > 0 );
        root.initializeOptimalSplitSearch( featuresToConsider );

        // Register all points with their respective parent nodes.
        for ( DataPointID pointID( 0 ), end( pointParents.size() ); pointID < end; ++pointID )
        {
            pointParents[pointID] = pointParents[pointID]->registerPoint( pointID, dataSet );
        }

        // Traverse all data points once for each feature, in order, so the tree nodes can find the best possible split for them.
        for ( unsigned int featureID = 0; featureID < featureCount; ++featureID ) // TODO: random trees should not use all features.
        {
            // Tell the tree that traversal is starting for this feature.
            root.startFeatureTraversal( featureID, featureCount - featureID, m_coin );

            // Traverse all datapoints in order of this feature.
            for ( auto it( featureIndex.featureBegin( featureID ) ), end( featureIndex.featureEnd( featureID ) ); it != end; ++it )
            {
                // Let the parent node of the data point know that it is being traversed.
                auto &tuple = *it;
                auto featureValue = std::get<0>( tuple );
                auto label        = std::get<1>( tuple );
                auto pointID      = std::get<2>( tuple );
                pointParents[pointID]->visitPoint( pointID, featureValue, label );
            }
        }

        // Allow all leaf nodes to split, if necessary.
        auto splitStats = root.split( m_maxDepth );
        std::cout << splitStats.splitsMade << " splits made." << std::endl;

        // Decide whether it is meaningful to continue.
        // TODO: make stop criterium more subtle.
        if ( splitStats.pointsLeftToImprove == 0 )
        {
            std::cout << "No more room for improvement." << std::endl;
            break;
        }
    }

    // Return a stripped version of the training tree.
    return root.finalize();
}
