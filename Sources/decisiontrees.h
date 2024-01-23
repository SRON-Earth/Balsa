#ifndef DECISIONTREES_H
#define DECISIONTREES_H

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "datarepresentation.h"
#include "exceptions.h"

/**
 * The unique consecutive ID of a decision tree node.
 */
typedef unsigned int NodeID;

/**
 * A decision tree.
 */
template <typename FeatureType = double, typename LabelType = bool>
class DecisionTree
{
    static_assert( std::is_arithmetic<FeatureType>::value,
        "Feature type should be an integral or floating point type." );
    static_assert( std::is_same<LabelType, bool>::value, "Label type should be 'bool'." );

public:

    typedef std::shared_ptr<DecisionTree<FeatureType, LabelType>> SharedPointer;
    typedef std::shared_ptr<const DecisionTree<FeatureType, LabelType>> ConstSharedPointer;

    struct DecisionTreeNode
    {
        NodeID leftChildID;
        NodeID rightChildID;
        FeatureID splitFeatureID;
        FeatureType splitValue;
        LabelType label;
    };

    typedef typename std::vector<DecisionTreeNode>::const_iterator ConstIterator;

    /**
     * Construct an empty decision tree.
     */
    DecisionTree()
    {
    }

    /**
     * Construct a decision tree from a sequence of decision tree nodes.
     */
    template <typename T>
    DecisionTree( T first, T last )
    : m_nodes( first, last )
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

private:

    void dump( NodeID nodeID, unsigned int indent ) const;
    unsigned int getDepth( NodeID nodeID ) const;

    std::vector<DecisionTreeNode> m_nodes;
};

/**
 * Determine whether a decision tree node is a leaf node.
 */
template <typename FeatureType, typename LabelType>
inline bool isLeafNode( const typename DecisionTree<FeatureType, LabelType>::DecisionTreeNode & node )
{
    return node.leftChildID == 0;
}

// #############################################################################
// # DecisionTree implementation.
// #############################################################################

template <typename FeatureType, typename LabelType>
unsigned int DecisionTree<FeatureType, LabelType>::getDepth( NodeID nodeID ) const
{
    const DecisionTreeNode & node = m_nodes[nodeID];
    const unsigned int depthLeft  = ( node.leftChildID == 0 ) ? 0 : getDepth( node.leftChildID );
    const unsigned int depthRight = ( node.rightChildID == 0 ) ? 0 : getDepth( node.rightChildID );
    return 1 + std::max( depthLeft, depthRight );
}

template <typename FeatureType, typename LabelType>
void DecisionTree<FeatureType, LabelType>::dump( NodeID nodeID, unsigned int indent ) const
{
    auto tab = std::string( indent, ' ' );

    const DecisionTreeNode & node = m_nodes[nodeID];
    if ( node.leftChildID || node.rightChildID )
    {
        // Internal node.
        std::cout << tab << "Node #" << nodeID << " Feature #" << static_cast<unsigned int>( node.splitFeatureID )
                  << ", split value = " << std::setprecision( 17 ) << node.splitValue << std::endl;
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

#endif // DECISIONTREES_H
