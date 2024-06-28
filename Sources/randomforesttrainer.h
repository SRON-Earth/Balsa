#ifndef RANDOMFORESTTRAINER_H
#define RANDOMFORESTTRAINER_H

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <semaphore>
#include <thread>
#include <vector>

#include "datatypes.h"
#include "fileio.h"
#include "indexeddecisiontree.h"
#include "messagequeue.h"
#include "table.h"

namespace balsa
{

/**
 * Trains a random forest classifier on a set of datapoints and known labels.
 */
template <typename FeatureIterator = Table<double>::ConstIterator, typename LabelIterator = Table<Label>::ConstIterator>
class RandomForestTrainer
{
    /**
     * An in internal message object to distribute training jobs to worker threads.
     */
    class TrainingJob
    {
    public:

        typedef typename IndexedDecisionTree<FeatureIterator, LabelIterator>::SeedType SeedType;

        TrainingJob( FeatureIterator dataSet, const IndexedDecisionTree<FeatureIterator, LabelIterator> & sapling, SeedType seed, unsigned int maxDepth, bool stop ):
        m_dataSet( dataSet ),
        m_sapling( sapling ),
        m_seed( seed ),
        m_maxDepth( maxDepth ),
        m_stop( stop )
        {
        }

        FeatureIterator                                             m_dataSet;
        const IndexedDecisionTree<FeatureIterator, LabelIterator> & m_sapling;
        SeedType                                                    m_seed;
        unsigned int                                                m_maxDepth;
        bool                                                        m_stop;
    };

    typedef MessageQueue<TrainingJob>                                                                 JobQueue;
    typedef MessageQueue<typename IndexedDecisionTree<FeatureIterator, LabelIterator>::SharedPointer> JobResultQueue;

public:

    /**
     * Constructor.
     * \param outputFile Name of the model file that will be written.
     * \param featuresToConsider Number of features to consider when splitting a
     *  node. When a node is to be split, the specified number of features will
     *  be randomly selected from the set of all features, and the optimal
     *  location for the split will be determined using the selected features.
     *  If no valid split can be found, then features that were initially
     *  skipped will be considered as well. Effectively, this parameter sets
     *  the minimum number of features that will be considered. More features
     *  will be considered if necessary to find a valid split. If set to zero,
     *  the square root of the number of features will be used(rounded down).
     * \param max_depth Maximum distance from any node to the root of the tree.
     * \param min_purity Minimum Gini-purity to reach. When the purity of a node
     *  reaches this minimum, the node will not be split further. A minimum
     *  purity of 1.0 (the default) means nodes will be split until all
     *  remaining data points in a node have the same label. The minimum
     *  possible Gini-purity for any node in a classification problem with M
     *  labels is 1/M. Setting the minimum purity to this number or lower means
     *  no nodes will be split at all.
     * \param treeCount Number of decision trees that will be trained.
     * \param concurrent_trainers The maximum number of decision trees that will
     *  be trained concurrently.
     * \param single_precision If `true`, single precision (32-bit) floats will
     *  be used instead of double precision (64-bit) floats. This significantly
     *  reduces the amount of memory used during training, at the expense of
     *  precision.
     */
    RandomForestTrainer( BalsaFileWriter & fileWriter, unsigned int featuresToConsider = 0, unsigned maxDepth = std::numeric_limits<unsigned int>::max(), double minPurity = 1.0, unsigned int treeCount = 10, unsigned int concurrentTrainers = 10, bool writeGraphviz = false ):
    m_fileWriter( fileWriter ),
    m_featuresToConsider( featuresToConsider ),
    m_maxDepth( maxDepth ),
    m_minPurity( minPurity ),
    m_treeCount( treeCount ),
    m_trainerCount( concurrentTrainers ),
    m_writeGraphviz( writeGraphviz )
    {
        // Ensure the specified minimum purity is in range.
        if ( m_minPurity < 0.0 || m_minPurity > 1.0 )
            throw ClientError( "The specified minimum purity is out of range [0.0, 1.0]." );
    }

    /**
     * Destructor.
     */
    virtual ~RandomForestTrainer()
    {
    }

    /**
     * Train a forest of random trees on the data. Results will be written to the current output file (see Constructor).
     */
    void train( FeatureIterator pointsStart, FeatureIterator pointsEnd, unsigned int featureCount, LabelIterator labelsStart )
    {
        // Check precionditions, etc.
        if ( featureCount == 0 ) throw ClientError( "Data points must have at least one feature." );
        auto dataset    = pointsStart;
        auto labels     = labelsStart;
        auto entryCount = std::distance( pointsStart, pointsEnd );
        if ( entryCount % featureCount ) throw ClientError( "Malformed dataset." );
        auto pointCount = entryCount / featureCount;

        // Determine the number of features to consider during each randomized split. If the supplied value was 0, default to floor(sqrt(featurecount)).
        unsigned int featuresToConsider = m_featuresToConsider ? m_featuresToConsider : std::floor( std::sqrt( featureCount ) );
        if ( featuresToConsider > featureCount ) throw ClientError( "The specified number of features to consider exceeds the number of features in the dataset." );

        // Determine the impurity treshold from the specified minimum purity.
        assert( m_minPurity >= 0.0 && m_minPurity <= 1.0 );
        double impurityTreshold = 1.0 - m_minPurity;

        // Create an indexed tree with only one node. This is expensive to build, so it is shared for copying between threads.
        IndexedDecisionTree<FeatureIterator, LabelIterator> sapling( dataset, labels, featureCount, pointCount, featuresToConsider, m_maxDepth, impurityTreshold );

        // Record the number of classes.
        unsigned int classCount = sapling.getClassCount();

        // Create message queues for communicating with the worker threads.
        JobQueue       jobOutbox;
        JobResultQueue treeInbox;

        // Start the worker threads.
        std::vector<std::thread> workers;
        for ( unsigned int i = 0; i < m_trainerCount; ++i )
        {
            workers.push_back( std::thread( &RandomForestTrainer::workerThread, &jobOutbox, &treeInbox ) );
        }

        // Create jobs for all trees.
        auto & seedSequence = getMasterSeedSequence();
        for ( unsigned int i = 0; i < m_treeCount; ++i ) jobOutbox.send( TrainingJob( dataset, sapling, seedSequence.next(), m_maxDepth, false ) );

        // Create 'stop' messages for all threads, to be picked up after all the work is done.
        for ( unsigned int i = 0; i < workers.size(); ++i ) jobOutbox.send( TrainingJob( dataset, sapling, 0, 0, true ) );

        // Create a forest model file and write the forest start marker and
        // header.
        typedef typename IndexedDecisionTree<FeatureIterator, LabelIterator>::FeatureType FeatureType;
        m_fileWriter.enterForest<FeatureType>( classCount, featureCount );

        // Wait for all the trees to come in, and write each tree to a forest file.
        for ( unsigned int i = 0; i < m_treeCount; ++i )
        {
            // Pull a tree from the inbox.
            auto tree = treeInbox.receive();

            // Write the tree without the bulky index, which is no longer needed
            // after training.
            typedef typename IndexedDecisionTree<FeatureIterator, LabelIterator>::LabelType LabelType;
            auto strippedTree = tree->template getDecisionTree<FeatureType *, LabelType *>();
            m_fileWriter.writeTree( *strippedTree );

            // Write a Graphviz file for the tree, if necessary.
            if ( m_writeGraphviz )
            {
                std::stringstream ss;
                ss << "tree" << i << ".dot";
                tree->writeGraphviz( ss.str() );
            }
        }

        // Write the forest end marker.
        m_fileWriter.leaveForest();

        // Wait for all the threads to join.
        for ( auto & worker : workers ) worker.join();
    }

private:

    static void workerThread( JobQueue * jobInbox, JobResultQueue * treeOutbox )
    {
        // Train trees until it is time to stop.
        while ( true )
        {
            // Get an assignment or stop message from the queue.
            TrainingJob job = jobInbox->receive();
            if ( job.m_stop ) break;

            // Clone the sapling and grow it. Take care to re-seed the random
            // generator used for feature selection, otherwise identical trees
            // will be grown.
            typename IndexedDecisionTree<FeatureIterator, LabelIterator>::SharedPointer tree( new IndexedDecisionTree<FeatureIterator, LabelIterator>( job.m_sapling ) );
            tree->seed( job.m_seed );
            tree->grow();
            treeOutbox->send( tree );
        }
    }

    BalsaFileWriter & m_fileWriter;
    unsigned int m_featuresToConsider;
    unsigned int m_maxDepth;
    double       m_minPurity;
    unsigned int m_treeCount;
    unsigned int m_trainerCount;
    bool         m_writeGraphviz;
};

} // namespace balsa

#endif // RANDOMFORESTTRAINER_H
