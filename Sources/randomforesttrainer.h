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

#include "datarepresentation.h"
#include "decisiontrees.h"
#include "messagequeue.h"
#include "serdes.h"
#include "trainers.h"
#include "utilities.h"

/**
 * Trains a random binary forest classifier on a TrainingDataSet.
 */
class BinaryRandomForestTrainer
{

    // Used for distributing jobs to threads.
    class TrainingJob
    {
    public:

        TrainingJob( const TrainingDataSet & dataSet,
            const FeatureIndex & featureIndex,
            unsigned int maxDepth,
            bool stop )
        : dataSet( dataSet )
        , featureIndex( featureIndex )
        , maxDepth( maxDepth )
        , stop( stop )
        {
        }

        const TrainingDataSet & dataSet;
        const FeatureIndex & featureIndex;
        unsigned int maxDepth;
        bool stop;
    };

public:

    /**
     * Constructor.
     * \param outputFile Name of the model file that will be written.
     * \param dataset A const reference to a training dataset. Modifying the set after construction of the trainer
     * invalidates the trainer. \param concurrentTrainers The maximum number of trees that may be trained concurrently.
     */
    BinaryRandomForestTrainer( const std::string & outputFile,
        unsigned maxDepth               = std::numeric_limits<unsigned int>::max(),
        unsigned int treeCount          = 10,
        unsigned int concurrentTrainers = 10 )
    : m_outputFile( outputFile )
    , m_maxDepth( maxDepth )
    , m_trainerCount( concurrentTrainers )
    , m_treeCount( treeCount )
    {
    }

    /**
     * Destructor.
     */
    virtual ~BinaryRandomForestTrainer()
    {
    }

    /**
     * Train a forest of random trees on the data. Results will be written to the current output file (see Constructor).
     */
    void train( TrainingDataSet::ConstSharedPointer dataset )
    {
        // Build the feature index that is common to all threads.
        FeatureIndex featureIndex( *dataset );

        // Create message queues for communicating with the worker threads.
        MessageQueue<TrainingJob> jobOutbox;
        MessageQueue<DecisionTree<>::SharedPointer> treeInbox;

        // Start the worker threads.
        std::vector<std::thread> workers;
        for ( unsigned int i = 0; i < m_trainerCount; ++i )
        {
            workers.push_back( std::thread( &BinaryRandomForestTrainer::workerThread, i, &jobOutbox, &treeInbox ) );
        }

        // Create jobs for all trees.
        for ( unsigned int i = 0; i < m_treeCount; ++i )
            jobOutbox.send( TrainingJob( *dataset, featureIndex, m_maxDepth, false ) );

        // Create 'stop' messages for all threads, to be picked up after all the work is done.
        for ( unsigned int i = 0; i < workers.size(); ++i )
            jobOutbox.send( TrainingJob( *dataset, featureIndex, 0, true ) );

        // Create a forest model file and write the 'f' header marker.
        std::ofstream out( m_outputFile, std::ios::binary | std::ios::out );
        serialize<char>( out, 'f' );

        // Wait for all the trees to come in, and write each tree to a forest file.
        for ( unsigned int i = 0; i < m_treeCount; ++i )
        {
            std::cout << "Tree #" << i << " completed." << std::endl;
            DecisionTree<>::SharedPointer tree = treeInbox.receive();
            serialize<char>( out, 't' );
            serialize<std::uint8_t>( out, dataset->getFeatureCount() );
            serialize<std::uint64_t>( out, tree->getNodeCount() );
            for ( const auto & node : *tree ) serialize<std::uint32_t>( out, node.leftChildID );
            for ( const auto & node : *tree ) serialize<std::uint32_t>( out, node.rightChildID );
            for ( const auto & node : *tree ) serialize<std::uint8_t>( out, node.splitFeatureID );
            for ( const auto & node : *tree ) serialize<double>( out, node.splitValue );
            for ( const auto & node : *tree ) serialize<bool>( out, node.label );
        }
        out.close();

        // Wait for all the threads to join.
        for ( auto & worker : workers ) worker.join();
    }

private:

    static void workerThread( unsigned int workerID,
        MessageQueue<TrainingJob> * jobInbox,
        MessageQueue<DecisionTree<>::SharedPointer> * treeOutbox )
    {
        // Train trees until it is time to stop.
        unsigned int jobsPickedUp = 0;
        while ( true )
        {
            // Get an assignment or stop message from the queue.
            TrainingJob job = jobInbox->receive();
            if ( job.stop ) break;
            ++jobsPickedUp;
            std::cout << "Worker #" << workerID << ": job " << jobsPickedUp << " picked up." << std::endl;

            // Train a tree and send it to the main thread.
            SingleTreeTrainerMark2 trainer( job.maxDepth );
            treeOutbox->send( trainer.train( job.featureIndex, job.dataSet ) );
            std::cout << "Worker #" << workerID << ": job " << jobsPickedUp << " finished." << std::endl;
        }

        std::cout << "Worker #" << workerID << " finished." << std::endl;
    }

    std::string m_outputFile;
    unsigned int m_maxDepth;
    unsigned int m_trainerCount;
    unsigned int m_treeCount;
};

#endif // RANDOMFORESTTRAINER_H
