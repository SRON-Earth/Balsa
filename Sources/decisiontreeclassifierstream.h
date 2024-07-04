#ifndef DECISIONTREECLASSIFIERSTREAM_H
#define DECISIONTREECLASSIFIERSTREAM_H

#include <fstream>

#include "classifierstream.h"
#include "decisiontreeclassifier.h"
#include "exceptions.h"
#include "fileio.h"

namespace balsa
{

/**
 * A classifier input stream implementation for random forests that loads decision
 * trees on demand.
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
template <typename FeatureIterator, typename OutputIterator>
class DecisionTreeClassifierInputStream: public ClassifierInputStream
{
public:

    using typename ClassifierInputStream<FeatureIterator, OutputIterator>::ClassifierType;

    typedef DecisionTreeClassifier<FeatureIterator, OutputIterator> DecisionTreeClassifierType;

    DecisionTreeClassifierInputStream( const std::string & filename, unsigned int maxPreload = 0 ):
    m_fileParser( filename ),
    m_maxPreload( maxPreload ),
    m_cacheIndex( 0 )
    {
        ForestHeader header = m_fileParser.enterForest();
        m_classCount = header.classCount;
    }

    DecisionTreeClassifierInputStream( const DecisionTreeClassifierInputStream & ) = delete;

    /**
     * Return the number of classes distinguished by the classifiers in this
     * stream.
     */
    unsigned int getClassCount() const
    {
        return m_classCount;
    }

    /**
     * Rewind the stream to the beginning.
     */
    void rewind()
    {
        // Flush the cache, unless *all* trees are being kept in memory.
        if ( m_maxPreload != 0 )
        {
            m_cache.clear();
        }

        // Reset the index of the next tree to the start of the cache.
        m_cacheIndex = 0;

        // Seek to the offset of the first decision tree in the model file.
        m_fileParser.reenterForest();
    }

    /**
     * Return the next classifier in the stream, or an empty shared pointer when
     * the end of the stream has been reached.
     */
    typename ClassifierType::SharedPointer next()
    {
        // Fetch more trees if necessary.
        if ( m_cacheIndex == m_cache.size() )
        {
            if ( m_maxPreload != 0 || m_cache.empty() )
            {
                fetch();
            }
        }

        // Return the next tree from the stream, or and empty shared pointer if
        // there are no more trees.
        if ( m_cacheIndex == m_cache.size() )
        {
            return typename ClassifierType::SharedPointer();
        }
        else
        {
            return m_cache[m_cacheIndex++];
        }
    }

private:

    void fetch()
    {
        // Flush the cache and reset the index of the next tree to the start of
        // the cache.
        m_cache.clear();
        m_cacheIndex = 0;

        // If the model file contains no more trees, do nothing.
        if ( !m_fileParser.atTree() )
        {
            return;
        }

        // Read decision trees into the cache, up to the maximum number of trees
        // to preload.
        while ( m_maxPreload == 0 || m_cache.size() < m_maxPreload )
        {
            if ( !m_fileParser.atTree() )
            {
                break;
            }

            auto tree = m_fileParser.parseTree<FeatureIterator, OutputIterator>();
            m_cache.push_back( tree );
        }
    }

    BalsaFileParser                                     m_fileParser;
    std::size_t                                         m_maxPreload;
    unsigned int                                        m_classCount;
    std::size_t                                         m_cacheIndex;
    std::vector<typename ClassifierType::SharedPointer> m_cache;
};

} // namespace balsa

#endif
