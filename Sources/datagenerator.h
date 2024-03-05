#ifndef DATAGENERATOR_H
#define DATAGENERATOR_H

#include <random>
#include <memory>
#include <iostream>
#include <string>

#include "genericparser.h"
#include "datatypes.h"
#include "table.h"

template<typename FeatureType>
class DataGenerator
{
  public:

    typedef std::shared_ptr<DataGenerator> SharedPointer;

  DataGenerator( unsigned int )
  {
  }

  virtual ~DataGenerator()
  {
  }

  virtual void generate( unsigned int pointCount, Table<FeatureType> &points, Table<Label> &labels ) = 0;

};

template<typename FeatureType>
class NumberGenerator
{
public:

  typedef std::mt19937 RandomEngine;
  typedef std::shared_ptr< NumberGenerator<FeatureType> > SharedPointer;

  virtual ~NumberGenerator()
  {
  }

  virtual FeatureType generate( RandomEngine &engine ) = 0;
};

template<typename FeatureType>
class GaussianNumberGenerator: public NumberGenerator<FeatureType>
{
public:

  GaussianNumberGenerator( FeatureType mean, FeatureType standardDeviation ):
  m_distribution( mean, standardDeviation )
  {
  }

  FeatureType generate( NumberGenerator<FeatureType>::RandomEngine &generator )
  {
      return m_distribution( generator );
  }

private:

  std::normal_distribution<FeatureType> m_distribution;
};

template<typename FeatureType>
class UniformNumberGenerator: public NumberGenerator<FeatureType>
{
public:

  UniformNumberGenerator( FeatureType lowerBound, FeatureType upperBound ):
  m_distribution( lowerBound, upperBound )
  {
  }

  FeatureType generate( NumberGenerator<FeatureType>::RandomEngine &generator )
  {
      return m_distribution( generator );
  }

private:

  std::uniform_real_distribution<FeatureType> m_distribution;
};

template<typename FeatureType>
class SingleSourceGenerator
{
public:

  typedef std::shared_ptr<SingleSourceGenerator<FeatureType> > SharedPointer;

  SingleSourceGenerator()
  {
  }

  void addFeatureGenerator( NumberGenerator<FeatureType>::SharedPointer generator )
  {
      m_features.push_back( generator );
  }

  void generatePoint( NumberGenerator<FeatureType>::RandomEngine &engine, Table<FeatureType> &points, unsigned int point )
  {
      // Generate data for the point.
      assert( points.getColumnCount() == m_features.size() );
      for ( unsigned int feature = 0; feature < points.getColumnCount(); ++feature )
      {
          points( point, feature ) = m_features[feature]->generate( engine );
      }
  }

private:

  std::vector< typename NumberGenerator<FeatureType>::SharedPointer > m_features;

};

template<typename FeatureType>
class MultiSourceGenerator: public DataGenerator<FeatureType>
{
public:

  typedef std::shared_ptr<MultiSourceGenerator<FeatureType>> SharedPointer;

  MultiSourceGenerator( unsigned int seed, unsigned int featureCount ):
  DataGenerator<FeatureType>( seed ),
  m_featureCount( featureCount ),
  m_engine( seed )
  {
  }

  virtual void generate( unsigned int pointCount, Table<FeatureType> &points, Table<Label> &labels )
  {
      // Create the tables.
      points = Table<FeatureType>( pointCount, m_featureCount );
      labels = Table<Label>      ( pointCount, 1              );

      // Generate the data.
      for ( unsigned int point = 0; point < pointCount; ++point )
      {
          // Select a label (ground truth).
          Label label = m_sourceDistribution( m_engine );
          assert( label < m_featureCount );
          labels( point, 0 ) = label;

          // Let the source of that label generate the data.
          m_sources[label]->generatePoint( m_engine, points, point );
      }
  }

  void addSource( FeatureType relativeFrequency, SingleSourceGenerator<FeatureType>::SharedPointer source )
  {
      // Add the source to the list of sources.
      m_sources.push_back( source );
      m_frequencies.push_back( relativeFrequency );

      // Create a distribution to select the sources.
      std::vector<FeatureType> indices( m_sources.size() );
      std::iota( indices.begin(), indices.end(), 0 );
      m_sourceDistribution = std::piecewise_constant_distribution<>( indices.begin(), indices.end(), m_frequencies.begin() );
  }

private:

  unsigned int m_featureCount;
  NumberGenerator<FeatureType>::RandomEngine m_engine;
  std::vector<typename SingleSourceGenerator<FeatureType>::SharedPointer> m_sources;
  std::vector<FeatureType> m_frequencies;
  std::piecewise_constant_distribution<FeatureType> m_sourceDistribution;

};

namespace
{

};

/**
 * Parse a data generator from a configuration file.
 */
template<typename FeatureType>
DataGenerator<FeatureType>::SharedPointer parseDataGenerator( std::istream &in, unsigned int seed = 0 )
{
    // Parse the datasource type name.
    GenericParser parser( in );
    auto datasourceType = parser.parseIdentifier();
    if ( datasourceType == "multisource" )
    {
        // Parse the parameters of the multisource definition.
        parser.consume( '(' );
        auto featureCount = parser.parseValue<unsigned int>();
        parser.consume( ')' );

        // Create an empty multisource.
        auto multisource = typename MultiSourceGenerator<FeatureType>::SharedPointer( new MultiSourceGenerator<FeatureType>( seed, featureCount ) );

        // Parse the body containing the sources.
        parser.consume( '{' );
        parser.consumeWhitespace();
        while ( parser.peek() != '}' )
        {
            // Parse a source definition.
            parser.consume( "source" );
            parser.consume( '(' );
            auto relativeFrequency = parser.parseValue<double>();
            parser.consume( ')' );
            parser.consume( "{"      );

            // Parse the feature generators.
            typename SingleSourceGenerator<FeatureType>::SharedPointer source( new SingleSourceGenerator<FeatureType>() );
            for ( unsigned int feature = 0; feature < featureCount; ++feature )
            {
                // Parse the type of the distribution.
                parser.consume( "feature" );
                parser.consume( '=' );
                auto distributionType = parser.parseIdentifier();
                parser.consume( '(' );
                if ( distributionType == "uniform" )
                {
                    // Add a generator to the source.
                    auto lowerBound = parser.parseValue<double>();
                    parser.consume( ',' );
                    auto upperBound = parser.parseValue<double>();
                    source->addFeatureGenerator( typename NumberGenerator<FeatureType>::SharedPointer( new UniformNumberGenerator( lowerBound, upperBound ) ) );
                }
                else if ( distributionType == "gaussian" )
                {
                    // Add a generator to the source.
                    auto mean = parser.parseValue<double>();
                    parser.consume( ',' );
                    auto standardDeviation = parser.parseValue<double>();
                    source->addFeatureGenerator( typename NumberGenerator<FeatureType>::SharedPointer( new GaussianNumberGenerator( mean, standardDeviation ) ) );

                }
                else throw ParseError( "Unrecognized random distribution type: " + distributionType );
                parser.consume( ')' );
                parser.consume( ';' );
            }

            // Add the source to the multisource.
            multisource->addSource( relativeFrequency, source );

            // Eat whitespace until the next 'source' or the closing bracket.
            parser.consumeWhitespace();

        }
        parser.consume( '}' );

        // Return the generator.
        return multisource;
    }

    throw ParseError( "Unrecognized data source definition: " + datasourceType );
}

#endif // DATAGENERATOR_H
