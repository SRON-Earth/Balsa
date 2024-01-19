#ifndef DECISIONTREES_H
#define DECISIONTREES_H

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

#include "datarepresentation.h"
#include "exceptions.h"

typedef unsigned char FeatureID;
typedef unsigned int NodeID;


/**
 * A decision tree.
 */
template <typename FeatureValueType = double, typename LabelValueType = unsigned char>
class DecisionTree
{
public:
  typedef std::shared_ptr<DecisionTree<FeatureValueType, LabelValueType>> SharedPointer;
  typedef std::shared_ptr<const DecisionTree<FeatureValueType, LabelValueType>> ConstSharedPointer;

  struct DecisionTreeNode
  {
      NodeID           leftChildID   ;
      NodeID           rightChildID  ;
      FeatureID        splitFeatureID;
      FeatureValueType splitValue    ;
      LabelValueType   label         ;
  };

  typedef typename std::vector<DecisionTreeNode>::const_iterator ConstIterator;

  /**
   * Construct an empty decision tree.
   */
  explicit DecisionTree( unsigned int featureCount )
  : m_featureCount( featureCount )
  {
  }

  /**
   * Construct a decision tree from a sequence of decision tree nodes.
   */
  template <typename T>
  DecisionTree( unsigned int featureCount, T first, T last )
  : m_featureCount( featureCount )
  , m_nodes( first, last )
  {
  }

  unsigned int getFeatureCount() const
  {
      return m_featureCount;
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

  unsigned int m_featureCount;
  std::vector<DecisionTreeNode> m_nodes;
};

/**
 * Determine whether a decision tree node is a leaf node.
 */
template <typename FeatureValueType, typename LabelValueType>
inline bool isLeafNode( const typename DecisionTree<FeatureValueType, LabelValueType>::DecisionTreeNode & node )
{
    return node.leftChildID == 0;
}

/**
 * Serialize a POD (Plain Old Data) value to a binary input stream.
 */
template <typename T>
void writePODValue( std::ostream & os, const T & value )
{
    os.write( reinterpret_cast<const char *>( &value ), sizeof( T ) );
}

template <typename LabelType>
void writeLabel( std::ostream & os, LabelType label );

template <>
void writeLabel( std::ostream & os, bool label );

template <>
void writeLabel( std::ostream & os, std::uint8_t label );

template <typename FeatureType>
void writeSplitValue( std::ostream & os, FeatureType splitValue );

template <>
void writeSplitValue( std::ostream & os, float splitValue );

template <>
void writeSplitValue( std::ostream & os, double splitValue );

/**
 * Write a decision tree to a binary output stream.
 */
template <typename FeatureValueType, typename LabelValueType>
void writeDecisionTree( std::ostream & os, const DecisionTree<FeatureValueType, LabelValueType> & tree )
{
    writePODValue<char>( os, 't' );
    writePODValue<std::uint8_t>( os, tree.getFeatureCount() );
    writePODValue<std::uint64_t>( os, tree.getNodeCount() );
    for ( auto const & node : tree )
    {
        writePODValue<std::uint32_t>( os, node.leftChildID );
    }
    for ( auto const & node : tree )
    {
        writePODValue<std::uint32_t>( os, node.rightChildID );
    }
    for ( auto const & node : tree )
    {
        writePODValue<std::uint8_t>( os, node.splitFeatureID );
    }
    for ( auto const & node : tree )
    {
        writeSplitValue( os, node.splitValue );
    }
    for ( auto const & node : tree )
    {
        writeLabel( os, node.label );
    }
}

template <typename FeatureValueType, typename LabelValueType>
void DecisionTree<FeatureValueType, LabelValueType>::dump( NodeID nodeID, unsigned int indent ) const
{
    auto tab = std::string( indent, ' ' );

    const DecisionTreeNode & node = m_nodes[nodeID];
    if (node.leftChildID || node.rightChildID)
    {
        // Internal node.
        std::cout << tab << "Node #" << nodeID
                  << " Feature #" << static_cast<unsigned int>( node.splitFeatureID )
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

template <typename FeatureValueType, typename LabelValueType>
unsigned int DecisionTree<FeatureValueType, LabelValueType>::getDepth( NodeID nodeID ) const
{
    const DecisionTreeNode & node = m_nodes[nodeID];
    const unsigned int depthLeft  = ( node.leftChildID  == 0 ) ? 0 : getDepth( node.leftChildID  );
    const unsigned int depthRight = ( node.rightChildID == 0 ) ? 0 : getDepth( node.rightChildID );
    return 1 + std::max( depthLeft, depthRight );
}

#endif // DECISIONTREES_H
