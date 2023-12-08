#ifndef DECISIONTREES_H
#define DECISIONTREES_H

#include <algorithm>
#include <memory>
#include <vector>
#include <iostream>

#include "datarepresentation.h"

/**
 * A decision tree.
 */
class DecisionTree
{
public:

  typedef std::shared_ptr <      DecisionTree> SharedPointer     ;
  typedef std::shared_ptr <const DecisionTree> ConstSharedPointer;

  typedef unsigned int NodeID;

  struct DecisionTreeNode
  {
      unsigned int  leftChildID   ;
      unsigned int  rightChildID  ;
      unsigned char splitFeatureID;
      double        splitValue    ;
      bool          label         ;
  };

  typedef std::vector<DecisionTreeNode>::const_iterator ConstIterator;

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
      return m_nodes.at( nodeID );
  }

  /**
   * Return a const reference to the node with the specified ID.
   */
  const DecisionTreeNode & operator[]( unsigned int nodeID ) const
  {
      return m_nodes.at( nodeID );
  }

  /**
   * Classify the specified data point.
   */
  bool classify( const DataSet &dataSet, DataPointID pointID ) const
  {
      return classify( NodeID( 0 ), dataSet, pointID );
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
  bool classify( NodeID nodeID, const DataSet &dataSet, DataPointID pointID ) const;
  unsigned int getDepth( NodeID nodeID ) const;

  std::vector<DecisionTreeNode> m_nodes;
};

/**
 * An ensemble of (usually randomized) decision trees.
 */
class Forest
{
public:

  typedef std::shared_ptr<Forest> SharedPointer;
  typedef std::vector<DecisionTree::ConstSharedPointer>::const_iterator ConstIterator;

  /**
   * Return an iterator to the beginning of the collection of decision trees in
   * this forest.
   */
  ConstIterator begin() const
  {
      return m_trees.begin();
  }

  /**
   * Return an iterator to the end of the collection of decision trees in this
   * forest.
   */
  ConstIterator end() const
  {
      return m_trees.end();
  }

  /**
   * Returns the number of nodes in the largest tree.
   */
  unsigned int getMaximumNodeCount() const
  {
      unsigned int maximum = 0;
      for ( auto &tree: m_trees ) maximum = std::max( maximum, tree->getNodeCount() );
      return maximum;
  }

  /**
   * Returns the depth of the largest tree.
   */
  unsigned int getMaximumDepth() const
  {
      unsigned int maximum = 0;
      for ( auto &tree: m_trees ) maximum = std::max( maximum, tree->getDepth() );
      return maximum;
  }

  /**
   * Classify all points in a DataSet.
   */
  std::vector<bool> classify( const DataSet &dataSet ) // TODO: Make this an interator-based, container independent template.
  {
      std::vector<bool> labels( dataSet.size() );
      for ( std::size_t i = 0; i < dataSet.size(); ++i )
      {
        labels[i] = classify( dataSet, i );
      }
      return labels;
  }

  /**
   * Return the classification of a data point.
   */
  bool classify( const DataSet &dataSet, DataPointID pointID )
  {
      unsigned int trueCount = 0;
      for ( auto tree: m_trees ) if ( tree->classify( dataSet, pointID ) ) ++trueCount;
      return trueCount >= m_trees.size() / 2; // TODO: weighted voting based on quality of out-of-bag predictions of subtrees?
  }

  /**
   * Add a tree to the forest.
   * \param tree The root node of the tree.
   */
  void addTree( DecisionTree::ConstSharedPointer tree )
  {
      m_trees.push_back( tree );
  }

  /**
   * Print the forest for debugging purposes.
   */
  void dump() const
  {
      std::cout << "Forest:" << std::endl;
      unsigned int i = 0;
      std::string rule( 80, '-' );
      for ( auto const & tree: m_trees )
      {
          // Dump the tree with a header.
          std::cout << rule << std::endl;
          std::cout << "Tree #" << i << std::endl;
          std::cout << rule << std::endl;
          tree->dump();

          // Increment counter.
          ++i;
      }
      std::cout << rule << std::endl;
  }

  /**
   * Remove all trees from the forest.
   */
  void clear()
  {
      m_trees.clear();
  }

private:

  std::vector<DecisionTree::ConstSharedPointer> m_trees;
};

/**
 * Determine whether a decision tree node is a leaf node.
 */
inline bool isLeafNode( const DecisionTree::DecisionTreeNode & node )
{
    return node.leftChildID == 0;
}

/**
 * Read a decision tree from a binary input stream.
 */
DecisionTree::SharedPointer readDecisionTree( std::istream & in );

/**
 * Write a decision tree to a binary input stream.
 */
void writeDecisionTree( std::ostream & os, const DecisionTree & tree );

/**
 * Writes a decision tree to a binary file.
 */
void storeDecisionTree( const DecisionTree & tree, const std::string & filename );

/**
 * Reads a Forest from a binary input stream. See also: loadForest().
 */
Forest::SharedPointer readForest( std::istream &in );

/**
 * Writes a Forest to a binary output stream. See also: storeForest().
 */
void writeForest( std::ostream & out, const Forest &forest );

/**
 * Reads a Forest from a file.
 */
Forest::SharedPointer loadForest( const std::string &filename );

/**
 * Writes a Forest to a file.
 */
void storeForest( const Forest &forest, const std::string &filename );

#endif // DECISIONTREES_H
