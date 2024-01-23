#ifndef ENSEMBLECLASSIFIER_H
#define ENSEMBLECLASSIFIER_H

#include <thread>

#include "messagequeue.h"

template <typename FeatureIterator, typename OutputIterator>
class EnsembleClassifier: public Classifier<FeatureIterator, OutputIterator>
{

public:

    using typename Classifier<FeatureIterator, OutputIterator>::VoteTable;

    /**
     * Creates an ensemble classifier.
     * \param classifiers A resettable stream of classifiers to apply.
     * \param maxWorkerThreads The maximum number of threads that may be created in addition to the main thread.
     */
    EnsembleClassifier( std::size_t featureCount,
        ClassifierStream<FeatureIterator, OutputIterator> & classifiers,
        unsigned int maxWorkerThreads = 0 )
    : Classifier<FeatureIterator, OutputIterator>( featureCount )
    , m_maxWorkerThreads( maxWorkerThreads )
    , m_classifierStream( classifiers )
    {
    }

    void classify( FeatureIterator pointsStart, FeatureIterator pointsEnd, OutputIterator labels ) const
    {
        // Check the dimensions of the input data.
        auto rawFeatureCount = std::distance( pointsStart, pointsEnd );
        auto featureCount    = this->getFeatureCount();
        assert( rawFeatureCount > 0 );
        assert( ( rawFeatureCount % featureCount ) == 0 );

        // Create a table for the label votes.
        unsigned int pointCount = rawFeatureCount / featureCount;
        VoteTable voteCounts( typename VoteTable::value_type( 0 ), pointCount );

        // Let all classifiers vote on the point labels.
        auto voterCount = classifyAndVote( pointsStart, pointsEnd, voteCounts );

        // Generate the labels by majority voting.
        auto limit = voterCount / 2;
        for ( auto voteCount : voteCounts ) *labels++ = voteCount >= limit;
    }

    unsigned int classifyAndVote( FeatureIterator pointsStart, FeatureIterator pointsEnd, VoteTable & table ) const
    {
        // Dispatch to single- or multithreaded implementation.
        if ( m_maxWorkerThreads > 0 )
            return classifyAndVoteMultiThreaded( pointsStart, pointsEnd, table );
        else
            return classifyAndVoteSingleThreaded( pointsStart, pointsEnd, table );
    }

private:

    /**
     * A job for the worker thread.
     */
    class WorkerJob
    {
    public:

        WorkerJob( typename Classifier<FeatureIterator, OutputIterator>::ConstSharedPointer classifier )
        : m_classifier( classifier )
        {
        }

        // Pointer to the tree that must be applied. Null indicates the end of processing, causing the worker to finish.
        typename Classifier<FeatureIterator, OutputIterator>::ConstSharedPointer m_classifier;
    };

    /**
     * A thread that runs classifyAndVote on a thread-local vote table.
     */
    class WorkerThread
    {
    public:

        typedef std::shared_ptr<WorkerThread> SharedPointer;

        WorkerThread( MessageQueue<WorkerJob> & jobQueue,
            std::size_t featureCount,
            FeatureIterator pointsStart,
            FeatureIterator pointsEnd )
        : m_running( false )
        , m_jobQueue( jobQueue )
        , m_pointsStart( pointsStart )
        , m_pointsEnd( pointsEnd )
        {
            // Check the dimensions of the input data.
            auto rawFeatureCount = std::distance( pointsStart, pointsEnd );
            assert( rawFeatureCount > 0 );
            assert( ( rawFeatureCount % featureCount ) == 0 );

            // Create a table for the label votes.
            unsigned int pointCount = rawFeatureCount / featureCount;
            m_votes.resize( pointCount, 0 );
        }

        void start()
        {
            // Launch a thread to process incoming jobs from the job queue.
            assert( !m_running );
            m_running = true;
            m_thread  = std::thread( &EnsembleClassifier::WorkerThread::processJobs, this );
        }

        void join()
        {
            // Wait for the thread to join.
            if ( !m_running ) return;
            m_thread.join();
            m_running = false;
        }

        const VoteTable & getVotes() const
        {
            return m_votes;
        }

    private:

        void processJobs()
        {
            // Process incoming jobs until the null job is received.
            for ( WorkerJob job( m_jobQueue.receive() ); job.m_classifier; job = m_jobQueue.receive() )
            {
                // Let the classifier vote on the data. Accumulate votes in the thread-private vote table.
                job.m_classifier->classifyAndVote( m_pointsStart, m_pointsEnd, m_votes );
            }
        }

        bool m_running;
        MessageQueue<WorkerJob> & m_jobQueue;
        FeatureIterator m_pointsStart;
        FeatureIterator m_pointsEnd;
        VoteTable m_votes;
        std::thread m_thread;
    };

    unsigned int classifyAndVoteSingleThreaded( FeatureIterator pointsStart,
        FeatureIterator pointsEnd,
        VoteTable & table ) const
    {
        // Reset the stream of classifiers, and apply each classifier that comes out of it.
        m_classifierStream.rewind();
        unsigned int voterCount = 0;
        for ( auto classifier = m_classifierStream.next(); classifier;
              classifier      = m_classifierStream.next(), ++voterCount )
            classifier->classifyAndVote( pointsStart, pointsEnd, table );

        // Return the number of classifiers that have voted.
        return voterCount;
    }

    unsigned int classifyAndVoteMultiThreaded( FeatureIterator pointsStart,
        FeatureIterator pointsEnd,
        VoteTable & table ) const
    {
        // Reset the stream of classifiers.
        m_classifierStream.rewind();
        unsigned int voterCount = 0;

        // Create message queues for communicating with the worker threads.
        MessageQueue<WorkerJob> jobQueue;

        // Create the workers.
        std::vector<typename WorkerThread::SharedPointer> workers;
        for ( unsigned int i = 0; i < m_maxWorkerThreads; ++i )
            workers.push_back( typename WorkerThread::SharedPointer(
                new WorkerThread( jobQueue, this->getFeatureCount(), pointsStart, pointsEnd ) ) );

        // Start all the workers.
        for ( auto & worker : workers ) worker->start();

        // Reset the stream of classifiers, and apply each classifier that comes out of it.
        for ( auto classifier = m_classifierStream.next(); classifier;
              classifier      = m_classifierStream.next(), ++voterCount )
            jobQueue.send( WorkerJob( classifier ) );

        // Send stop messages for all workers.
        for ( auto i = workers.size(); i > 0; --i ) jobQueue.send( WorkerJob( nullptr ) );

        // Wait for all the workers to join.
        for ( auto & worker : workers ) worker->join();

        // Add the votes accumulated by the workers to the output total.
        for ( auto & worker : workers ) table += worker->getVotes();

        // Return the number of classifiers that have voted.
        return voterCount;
    }

    unsigned int m_maxWorkerThreads;
    ClassifierStream<FeatureIterator, OutputIterator> & m_classifierStream;
};

#endif // ENSEMBLECLASSIFIER_H
