#ifndef DECISIONTREECLASSIFIERSTREAM_H
#define DECISIONTREECLASSIFIERSTREAM_H

#include "classifierstream.h"
#include "decisiontrees.h"
#include "exceptions.h"

/**
 * Class that represents a collection of decision trees that can be iterated.
 */
template<typename FeatureIterator, typename OutputIterator>
class DecisionTreeClassifierStream: public ClassifierStream<FeatureIterator, OutputIterator>
{
public:
  using typename ClassifierStream<FeatureIterator, OutputIterator>::ClassifierType;

  DecisionTreeClassifierStream( const std::string &filename, unsigned int maxPreload = 0 )
  : m_filename( filename )
  , m_maxPreload( maxPreload )
  , m_cacheIndex( 0 )
  {
  }

  DecisionTreeClassifierStream( const DecisionTreeClassifierStream & ) = delete;

  /**
   * Rewind the stream to the beginning.
   */
  virtual void rewind()
  {
      // Flush the cache, unless *all* trees are being kept in memory.
      if ( m_maxPreload != 0 )
      {
          m_cache.clear();
      }

      // Reset the index of the next tree to the start of the cache.
      m_cacheIndex = 0;

      // Close the model file. This will cause it to be re-opened in fetch(),
      // thus re-starting the iteration over trees.
      m_modelFile.close();
  }

  /**
   * Return the next classifier in the stream, or an empty shared pointer when
   * the end of the stream has been reached.
   */
  virtual typename DecisionTreeClassifierStream::ClassifierType::SharedPointer next()
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

      // Re-open the model file if necessary.
      if ( !m_modelFile.is_open() )
      {
          // Open the model file.
          m_modelFile.open( m_filename );
          if ( !m_modelFile.is_open() )
          {
               throw SupplierError( "Unable to open model file." );
          }

          // Parse the model header. This will advance the file stream to the
          // first tree.
          char marker = 0;
          m_modelFile >> marker;
          if ( marker != 'f' )
          {
               throw ParseError( "Unexpected header block." );
          }
      }

      // If the model file contains no more trees, do nothing.
      if ( m_modelFile.eof() )
      {
          return;
      }

      // Read decision trees into the cache, up to the maximum number of trees
      // to preload.
      while ( m_maxPreload == 0 || m_cache.size() < m_maxPreload )
      {
          if ( m_modelFile.peek() == std::ifstream::traits_type::eof() )
          {
              break;
          }

          auto tree = readDecisionTree<FeatureIterator, OutputIterator>( m_modelFile );
          m_cache.push_back( tree );
      }

      // Raise an exception if an error occurred while reading.
      if ( m_modelFile.fail() )
      {
          throw SupplierError( "Error reading model file." );
      }
  }

  std::string   m_filename;
  std::size_t   m_maxPreload;
  std::ifstream m_modelFile;
  std::size_t   m_cacheIndex;
  std::vector<typename ClassifierType::SharedPointer> m_cache;
};

#endif
