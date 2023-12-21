#include <iostream>
#include <cassert>

#include "decisiontrees.h"


// Tests consistency between the brute force classifier and the fast bulk classifyVote() method of DecisionTree.
bool testClassifyVote( )
{

    // Construct a relatively simple, but non-trivial tree.
    DecisionTree tree;
    auto root = tree.addNode( DecisionTree::DecisionTreeNode() ); // Root.
    assert( root == 0 );
    auto l = tree.addNode( DecisionTree::DecisionTreeNode() );
    auto r = tree.addNode( DecisionTree::DecisionTreeNode() );
    tree[root].leftChildID  = l;
    tree[root].rightChildID = r;
    auto rl = tree.addNode( DecisionTree::DecisionTreeNode() );
    auto rr = tree.addNode( DecisionTree::DecisionTreeNode() );
    tree[r].leftChildID  = rl;
    tree[r].rightChildID = rr;
    tree[root].splitFeatureID = 3;
    tree[root].splitValue     = 5;
    tree[r].splitFeatureID = 0;
    tree[r].splitValue     = 3;

    tree[l].label  = false;
    tree[rl].label = true ;
    tree[rr].label = false;


    // Create a dataset.
    DataSet points( 4 );
    points.appendDataPoint( {10  , 10, 10, 4.9} ); // Should be false.
    points.appendDataPoint( { 2.9, 0 , 6,  5  } ); // Should be true.
    points.appendDataPoint( {10  , 0 , 6,  5  } ); // Shuuld be false.
    points.appendDataPoint( {3  , 0 , 6,  5   } ); // Should be false.

    // Classify the dataset using the bulk classification method.
    std::vector<unsigned int> votes( points.size(), 0 );
    tree.dump();
    tree.classifyVote( points, votes );

    // Check the classification against naive classification.
    bool success = true;
    for ( unsigned int point = 0; point < points.size(); ++point )
    {
        if ( tree.classify( points, point ) != ( votes[point] > 0 ) )
        {
            success = false;
            std::cout << "Inconsistency on point " << point << std::endl;
        }
    }

    return success;
}

int main( int, char ** )
{
    unsigned int successes = 0;
    unsigned int failures  = 0;
    if ( testClassifyVote() ) ++successes; else ++failures;

    std::cout << successes << " successes, " << failures << " failures." << std::endl;
    return failures;
}
