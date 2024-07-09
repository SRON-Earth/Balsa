#ifndef CLASSIFIERFILESTREAM_H
#define CLASSIFIERFILESTREAM_H

#include "classifierstream.h"
#include "fileio.h"

namespace balsa
{

/**
 * A classifier input stream implementation for that loads classifiers on demand.
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

    /*
     * Construct an open classifier input stream.
     *
     * The \c maxPreload parameter determines how many classifiers to preload
     * (cache) in memory. This allows a trade-off to be made between memory
     * usage and disk I/O.
     *
     * If set to zero, all classifiers present in the input file are loaded into
     * memory. Calling \c next() always return classifiers from memory, and
     * calling \c rewind() will not cause classifiers to be reloaded.
     *
     * If set to a positive value \c N, this determines the number of
     * classifiers that will be read from the input file and cached. Calls
     * to \c next() will return classifiers from the cache. If the cache is
     * empty, the next \c N classifiers will be read into the cache. Calling \c
     * rewind() will empty the cache and reposition the input stream at the
     * beginning.
     *
     * \param filename Name of the file to open.
     * \param maxPreload The number of classifiers to preload (cache).
     */
    ClassifierFileInputStream( const std::string & filename, unsigned int maxPreload = 0 ):
    m_fileParser( filename ),
    m_maxPreload( maxPreload ),
    m_cacheIndex( 0 )
    {
        EnsembleHeader header = m_fileParser.enterEnsemble();
        m_classCount          = header.classCount;
        m_featureCount        = header.featureCount;
    }

    /*
     * Copy constructor (deleted). Classifier input streams cannot be copied.
     */
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
        m_fileParser.reenterEnsemble();
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
     * Constructs an open ensemble output stream.
     *
     * \param filename Name of the file to write.
     * \param creatorName Name of the tool that created the file (optional).
     *  This information will be stored in the file header.
     * \param creatorMajorVersion Major version number of the tool that created
     *  the file (optional). This information will be stored in the file header.
     * \param creatorMinorVersion Minor version number of the tool that created
     *  the file (optional). This information will be stored in the file header.
     * \param creatorPatchVersion Patch version number of the tool that created
     *  the file (optional). This information will be stored in the file header.
     */
    EnsembleFileOutputStream( const std::string & filename,
        std::optional<std::string>                creatorName         = std::nullopt,
        std::optional<unsigned char>              creatorMajorVersion = std::nullopt,
        std::optional<unsigned char>              creatorMinorVersion = std::nullopt,
        std::optional<unsigned char>              creatorPatchVersion = std::nullopt ):
    m_fileWriter( filename, creatorName, creatorMajorVersion, creatorMinorVersion, creatorPatchVersion ),
    m_classCount( 0 ),
    m_featureCount( 0 )
    {
    }

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
        if ( m_classCount != 0 ) m_fileWriter.leaveEnsemble();
    }

    /**
     * Perform the actual write in a subclass-specific way.
     * This is guaranteed to be called only when the stream is still open.
     */
    void onWrite( const Classifier & classifier )
    {
        if ( m_classCount == 0 )
        {
            m_classCount   = classifier.getClassCount();
            m_featureCount = classifier.getFeatureCount();
            m_fileWriter.enterEnsemble( m_classCount, m_featureCount );
        }

        assert( classifier.getClassCount() == m_classCount );
        assert( classifier.getFeatureCount() == m_featureCount );
        m_fileWriter.writeClassifier( classifier );
    }

    BalsaFileWriter m_fileWriter;
    unsigned int    m_classCount;
    unsigned int    m_featureCount;
};

} // namespace balsa

#endif // CLASSIFIERFILESTREAM_H
