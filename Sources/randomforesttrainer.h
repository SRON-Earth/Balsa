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

#include "indexeddecisiontree.h"
#include "datatypes.h"
#include "table.h"
#include "messagequeue.h"
#include "serdes.h"

/**
 * Trains a random forest classifier on a set of datapoints and known labels.
 */
template <typename FeatureType = double>
class RandomForestTrainer
{
    /**
     * An in internal message object to distribute training jobs to worker threads.
     */
    class TrainingJob
    {
    public:

        typedef IndexedDecisionTree<FeatureType>::SeedType SeedType;

        TrainingJob( const Table<FeatureType> & dataSet, const IndexedDecisionTree<FeatureType> & sapling, SeedType seed, unsigned int maxDepth, bool stop ):
        m_dataSet( dataSet ),
        m_sapling( sapling ),
        m_seed( seed ),
        m_maxDepth( maxDepth ),
        m_stop( stop )
        {
        }

        const Table<FeatureType> &               m_dataSet;
        const IndexedDecisionTree<FeatureType> & m_sapling;
        SeedType                                 m_seed;
        unsigned int                             m_maxDepth;
        bool                                     m_stop;
    };

  typedef MessageQueue<TrainingJob>                                              JobQueue;
  typedef MessageQueue<typename IndexedDecisionTree<FeatureType>::SharedPointer> JobResultQueue;

public:

    /**
     * Constructor.
     * \param outputFile Name of the model file that will be written.
     * \param concurrentTrainers The maximum number of trees that may be trained concurrently.
     */
    RandomForestTrainer( const std::string & outputFile, unsigned maxDepth = std::numeric_limits<unsigned int>::max(), unsigned int treeCount = 10, unsigned int concurrentTrainers = 10, unsigned int featuresToScan = 0, bool writeGraphviz = false ):
    m_outputFile( outputFile ),
    m_maxDepth( maxDepth ),
    m_trainerCount( concurrentTrainers ),
    m_treeCount( treeCount ),
    m_featuresToScan( featuresToScan ),
    m_writeGraphviz( writeGraphviz )
    {
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
    void train( const Table<FeatureType> & dataset, const Table<Label> & labels )
    {
        // Determine the number of features to consider during each randomized split. If the supplied value was 0, default to floor(sqrt(featurecount)).
        unsigned int numberOfFeatures = dataset.getColumnCount();
        unsigned int featuresToConsider = m_featuresToScan ? m_featuresToScan : std::floor( std::sqrt( numberOfFeatures ) );
        if ( featuresToConsider > numberOfFeatures ) throw ClientError( "The supplied number of features to scan exceeds the number of features in the dataset." );

        // Create an indexed tree with only one node. This is expensive to build, so it is shared for copying between threads.
        IndexedDecisionTree sapling( dataset, labels, featuresToConsider, m_maxDepth );

        // Create message queues for communicating with the worker threads.
        JobQueue       jobOutbox;
        JobResultQueue treeInbox;

        // Start the worker threads.
        std::vector<std::thread> workers;
        for ( unsigned int i = 0; i < m_trainerCount; ++i )
        {
            workers.push_back( std::thread( &RandomForestTrainer::workerThread, i, &jobOutbox, &treeInbox ) );
        }

        // Create jobs for all trees.
        auto & seedSequence = getMasterSeedSequence();
        for ( unsigned int i = 0; i < m_treeCount; ++i ) jobOutbox.send( TrainingJob( dataset, sapling, seedSequence.next(), m_maxDepth, false ) );

        // Create 'stop' messages for all threads, to be picked up after all the work is done.
        for ( unsigned int i = 0; i < workers.size(); ++i ) jobOutbox.send( TrainingJob( dataset, sapling, 0, 0, true ) );

        // Create a forest model file and write the forest header marker ("frst").
        std::ofstream out( m_outputFile, std::ios::binary | std::ios::out );
        out.write( "frst", 4 );

        // Wait for all the trees to come in, and write each tree to a forest file.
        for ( unsigned int i = 0; i < m_treeCount; ++i )
        {
            std::cout << "Tree #" << i << " completed." << std::endl;
            auto tree = treeInbox.receive();

            // Write the tree without the bulky index, which is no longer needed after training.
            tree->writeDecisionTreeClassifier( out );

            // Write a Graphviz file for the tree, if necessary.
            if ( m_writeGraphviz )
            {
                std::stringstream ss;
                ss << "tree#" << i << ".dot";
                tree->writeGraphviz( ss.str() );
            }
        }
        out.close();

        // Wait for all the threads to join.
        for ( auto & worker : workers ) worker.join();
    }

private:

    static void workerThread( unsigned int workerID, JobQueue *jobInbox, JobResultQueue *treeOutbox )
    {
        // Train trees until it is time to stop.
        unsigned int jobsPickedUp = 0;
        while ( true )
        {
            // Get an assignment or stop message from the queue.
            TrainingJob job = jobInbox->receive();
            if ( job.m_stop ) break;
            ++jobsPickedUp;
            std::cout << "Worker #" << workerID << ": job " << jobsPickedUp << " picked up." << std::endl;

            // Clone the sapling and grow it. Take care to re-seed the random
            // generator used for feature selection, otherwise identical trees
            // will be grown.
            typename IndexedDecisionTree<FeatureType>::SharedPointer tree( new IndexedDecisionTree<FeatureType>( job.m_sapling ) );
            tree->seed( job.m_seed );
            tree->grow();
            treeOutbox->send( tree );
            std::cout << "Worker #" << workerID << ": job " << jobsPickedUp << " finished." << std::endl;
        }

        std::cout << "Worker #" << workerID << " finished." << std::endl;
    }

    std::string  m_outputFile;
    unsigned int m_maxDepth;
    unsigned int m_trainerCount;
    unsigned int m_treeCount;
    unsigned int m_featuresToScan;
    bool         m_writeGraphviz;
};

#endif // RANDOMFORESTTRAINER_H
