#include <iostream>
#include <cassert>

#include "decisiontrees.h"

// Tests classification on a relatively simple, but non-trivial decision tree.
bool testClassification()
{
    // Create a dataset.
    DataSet points( 4 );
    points.appendDataPoint( {10  , 10, 10, 4.9} ); // Should be false.
    points.appendDataPoint( { 2.9, 0 , 6,  5  } ); // Should be true.
    points.appendDataPoint( {10  , 0 , 6,  5  } ); // Shuuld be false.
    points.appendDataPoint( { 3  , 0 , 6,  5  } ); // Should be false.

    // Create expected labels.
    std::vector<bool> expectedLabels{false, true, false, false};

    // Allocate space for the classification result.
    std::vector<bool> labels( points.size(), false );

    // Construct a relatively simple, but non-trivial decision tree.
    typedef typename std::decay_t<decltype( points.getData() )>::const_iterator FeatureIteratorType;
    typedef typename decltype( labels )::iterator OutputIteratorType;
    typedef DecisionTree<FeatureIteratorType, OutputIteratorType> DecisionTreeType;
    typedef typename DecisionTreeType::DecisionTreeNode DecisionTreeNodeType;

    DecisionTreeType tree( points.getFeatureCount() );
    auto root = tree.addNode( DecisionTreeNodeType() ); // Root.
    assert( root == 0 );
    auto l = tree.addNode( DecisionTreeNodeType() );
    auto r = tree.addNode( DecisionTreeNodeType() );
    tree[root].leftChildID  = l;
    tree[root].rightChildID = r;
    auto rl = tree.addNode( DecisionTreeNodeType() );
    auto rr = tree.addNode( DecisionTreeNodeType() );
    tree[r].leftChildID  = rl;
    tree[r].rightChildID = rr;
    tree[root].splitFeatureID = 3;
    tree[root].splitValue     = 5;
    tree[r].splitFeatureID = 0;
    tree[r].splitValue     = 3;

    tree[l].label  = false;
    tree[rl].label = true ;
    tree[rr].label = false;

    // Classify the dataset using the bulk classification method.
    tree.dump();
    tree.classify( points.getData().begin(), points.getData().end(), labels.begin() );

    // Check the classification against the expected result.
    bool success = true;
    for ( unsigned int point = 0; point < points.size(); ++point )
    {
        if ( labels[point] != expectedLabels[point] )
        {
            success = false;
            std::cout << "Inconsistency on point #" << point << std::endl;
        }
    }

    return success;
}

int main( int, char ** )
{
    unsigned int successes = 0;
    unsigned int failures  = 0;
    if ( testClassification() ) ++successes; else ++failures;

    std::cout << successes << " successes, " << failures << " failures." << std::endl;
    return failures;
}
