#ifndef CLASSIFIERFILESTREAM_H
#define CLASSIFIERFILESTREAM_H

#include "classifierstream.h"
#include "fileio.h"

namespace balsa
{

/**
 * A classifier input stream implementation for random forests that loads
 * classifiers on demand.
 *
 * Loading classifiers on demand enables ensemble classification using a minimal
 * amount of memory. Only the classifiers from the ensemble that are being
 * evaluated need to be kept in memory simultaneously. In the single threaded
 * case, only a single classifier is kept in memory at any given time.
 *
 * When classifying datasets in batches, loading classifiers on demand is
 * inefficient, because each classifier in the stream will be (re)loaded for
 * each dataset in the batch. If enough memory is available, consider setting
 * `maxPreload` to zero. This will cause all classifiers to be loaded into
 * memory once.
 */
class ClassifierFileInputStream: public ClassifierInputStream
{
public:

    ClassifierFileInputStream( const std::string & filename, unsigned int maxPreload = 0 ):
    m_fileParser( filename ),
    m_maxPreload( maxPreload ),
    m_cacheIndex( 0 )
    {
        ForestHeader header = m_fileParser.enterForest();
        m_classCount = header.classCount;
        m_featureCount = header.featureCount;
    }

    ClassifierFileInputStream( const ClassifierFileInputStream & ) = delete;

    /**
     * Return the number of classes distinguished by the classifiers in this
     * stream.
     */
    unsigned int getClassCount() const
    {
        return m_classCount;
    }

    /**
     * Returns the number of features the classifier expects.
     */
    unsigned int getFeatureCount() const
    {
        return m_featureCount;
    }

    /**
     * Rewind the stream to the beginning.
     */
    void rewind()
    {
        // Flush the cache, unless *all* classifiers are being kept in memory.
        if ( m_maxPreload != 0 )
        {
            m_cache.clear();
        }

        // Reset the index of the next classifier to the start of the cache.
        m_cacheIndex = 0;

        // Seek to the offset of the first classifier in the model file.
        m_fileParser.reenterForest();
    }

    /**
     * Return the next classifier in the stream, or an empty shared pointer when
     * the end of the stream has been reached.
     */
    Classifier::SharedPointer next()
    {
        // Fetch more classifiers if necessary.
        if ( m_cacheIndex == m_cache.size() )
        {
            if ( m_maxPreload != 0 || m_cache.empty() )
            {
                fetch();
            }
        }

        // Return the next classifier from the stream, or and empty shared pointer if
        // there are no more classifiers.
        if ( m_cacheIndex == m_cache.size() )
        {
            return Classifier::SharedPointer();
        }
        else
        {
            return m_cache[m_cacheIndex++];
        }
    }

private:

    void fetch()
    {
        // Flush the cache and reset the index of the next classifier to the start of
        // the cache.
        m_cache.clear();
        m_cacheIndex = 0;

        // If the model file contains no more classifiers, do nothing.
        if ( !m_fileParser.atTree() )
        {
            return;
        }

        // Read classifiers into the cache, up to the maximum number of
        // classifiers to preload.
        while ( m_maxPreload == 0 || m_cache.size() < m_maxPreload )
        {
            if ( !m_fileParser.atTree() )
            {
                break;
            }

            auto classifier = m_fileParser.parseClassifier();
            m_cache.push_back( classifier );
        }
    }

    BalsaFileParser                        m_fileParser;
    std::size_t                            m_maxPreload;
    unsigned int                           m_classCount;
    unsigned int                           m_featureCount;
    std::size_t                            m_cacheIndex;
    std::vector<Classifier::SharedPointer> m_cache;
};

class EnsembleFileOutputStream: public ClassifierOutputStream
{
public:

    /**
    * Constructs an open stream.
    */
    EnsembleFileOutputStream( const std::string & filename ):
    ClassifierOutputStream(),
    m_fileWriter( filename ),
    m_classCount( 0 ),
    m_featureCount( 0 )
    {
    }

    // TODO: Why is this necessary??
    ~EnsembleFileOutputStream()
    {
        close();
    }

private:

    /**
    * Perform subclass-specific operations when the stream is closed.
    */
    void onClose()
    {
        m_fileWriter.leaveForest();
    }

    /**
    * Perform the actual write in a subclass-specific way.
    * This is guaranteed to be called only when the stream is still open.
    */
    void onWrite( const Classifier &classifier )
    {
        if ( m_classCount == 0 )
        {
            m_classCount = classifier.getClassCount();
            m_featureCount = classifier.getFeatureCount();
            m_fileWriter.enterForest( m_classCount, m_featureCount );
        }

        assert( classifier.getClassCount() == m_classCount );
        assert( classifier.getFeatureCount() == m_featureCount );
        m_fileWriter.writeClassifier( classifier );
    }

// private:

    BalsaFileWriter m_fileWriter;
    unsigned int    m_classCount;
    unsigned int    m_featureCount;

};

} // namespace balsa

#endif // CLASSIFIERFILESTREAM_H
