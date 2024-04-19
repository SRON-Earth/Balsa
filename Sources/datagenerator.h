#ifndef DATAGENERATOR_H
#define DATAGENERATOR_H

#include <array>
#include <iostream>
#include <memory>
#include <random>
#include <string>

#include "datatypes.h"
#include "exceptions.h"
#include "genericparser.h"
#include "table.h"

namespace balsa
{

template <typename FeatureType>
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

    virtual void generate( unsigned int pointCount, Table<FeatureType> & points, Table<Label> & labels ) = 0;
};

template <typename FeatureType>
class FeatureGenerator
{
public:

    typedef std::mt19937                                   RandomEngine;
    typedef std::shared_ptr<FeatureGenerator<FeatureType>> SharedPointer;

    virtual ~FeatureGenerator()
    {
    }

    virtual unsigned int getFeatureCount() const = 0;

    template <typename FeatureIterator>
    FeatureIterator generate( RandomEngine & engine, FeatureIterator featureIterator )
    {
        const FeatureType * first = generate( engine );
        for ( const FeatureType * last = first + getFeatureCount(); first != last; )
        {
            *featureIterator++ = *first++;
        }
        return featureIterator;
    }

protected:

    virtual const FeatureType * generate( RandomEngine & engine ) = 0;
};

template <typename FeatureType>
class UniformFeatureGenerator: public FeatureGenerator<FeatureType>
{
public:

    typedef std::shared_ptr<UniformFeatureGenerator<FeatureType>> SharedPointer;

    UniformFeatureGenerator( FeatureType lowerBound, FeatureType upperBound ):
    m_distribution( lowerBound, upperBound )
    {
    }

    unsigned int getFeatureCount() const
    {
        return m_value.size();
    }

protected:

    const FeatureType * generate( FeatureGenerator<FeatureType>::RandomEngine & engine )
    {
        m_value[0] = m_distribution( engine );
        return m_value.data();
    }

private:

    std::uniform_real_distribution<FeatureType> m_distribution;
    std::array<FeatureType, 1>                  m_value;
};

template <typename FeatureType>
class GaussianFeatureGenerator: public FeatureGenerator<FeatureType>
{
public:

    typedef std::shared_ptr<GaussianFeatureGenerator<FeatureType>> SharedPointer;

    GaussianFeatureGenerator( FeatureType mean, FeatureType standardDeviation ):
    m_distribution( mean, standardDeviation )
    {
    }

    unsigned int getFeatureCount() const
    {
        return m_value.size();
    }

protected:

    const FeatureType * generate( FeatureGenerator<FeatureType>::RandomEngine & engine )
    {
        m_value[0] = m_distribution( engine );
        return m_value.data();
    }

private:

    std::normal_distribution<FeatureType> m_distribution;
    std::array<FeatureType, 1>            m_value;
};

template <typename FeatureType>
class AnnulusFeatureGenerator: public FeatureGenerator<FeatureType>
{
public:

    typedef std::shared_ptr<AnnulusFeatureGenerator<FeatureType>> SharedPointer;

    AnnulusFeatureGenerator( FeatureType minRadius, FeatureType maxRadius ):
    m_radiusDistribution( minRadius, maxRadius ),
    m_angleDistribution( 0.0, 2.0 * std::numbers::pi )
    {
        assert( minRadius >= 0.0 && maxRadius > minRadius );
    }

    unsigned int getFeatureCount() const
    {
        return m_value.size();
    }

protected:

    const FeatureType * generate( FeatureGenerator<FeatureType>::RandomEngine & engine )
    {
        FeatureType radius = m_radiusDistribution( engine );
        FeatureType angle = m_angleDistribution( engine );
        m_value[0] = radius * std::cos( angle );
        m_value[1] = radius * std::sin( angle );
        return m_value.data();
    }

private:

    std::uniform_real_distribution<FeatureType> m_radiusDistribution;
    std::uniform_real_distribution<FeatureType> m_angleDistribution;
    std::array<FeatureType, 2>                  m_value;
};

template <typename FeatureType>
class CheckerboardFeatureGenerator: public FeatureGenerator<FeatureType>
{
public:

    typedef std::shared_ptr<CheckerboardFeatureGenerator<FeatureType>> SharedPointer;

    enum class Color
    {
        BLACK,
        WHITE
    };

    CheckerboardFeatureGenerator( Color color ):
    m_color( color )
    {
    }

    void addDimension( unsigned int cellCount, FeatureType cellSize )
    {
        m_cellSize.push_back( cellSize );
        m_cellCount.push_back( cellCount );
        m_distribution.emplace_back( 0.0, cellCount * cellSize );
        m_value.push_back( FeatureType() );
    }

    unsigned int getFeatureCount() const
    {
        return m_value.size();
    }

protected:

    const FeatureType * generate( FeatureGenerator<FeatureType>::RandomEngine & engine )
    {
        // Randomly draw points inside the checkerboard; stop when a point is
        // found that is located on a cell of the selected color. Since half of
        // all possible points is locate on a cell of the selected color, the
        // expected number of times a point has to be drawn is equal to two.
        while ( true )
        {
            // Randomly draw a point and maintain the sum of the coordinates of
            // the cell on which the point is located.
            unsigned int sum = 0;
            for ( unsigned int i = 0; i < m_value.size(); ++i )
            {
                // Draw a coordinate along the current axis.
                const FeatureType coordinate = m_distribution[i]( engine );
                m_value[i] = coordinate;

                // Update the sum of cell coordinates.
                const FeatureType cellCoordinate = std::floor( coordinate / m_cellSize[i] );
                assert( cellCoordinate >= 0.0 );
                sum += static_cast<unsigned int>( cellCoordinate );
            }

            // The sum of the coordinates of a cell is even iff the cell is
            // black.
            if ( ( sum % 2 == 0 ) == ( m_color == Color::BLACK ) )
            {
                break;
            }
        }

        // Center the checkerboard around the origin.
        for ( unsigned int i = 0; i < m_value.size(); ++i )
        {
            m_value[i] -= m_cellSize[i] / 2.0 * m_cellCount[i];
        }

        return m_value.data();
    }

private:

    Color                                                    m_color;
    std::vector<FeatureType>                                 m_cellSize;
    std::vector<unsigned int>                                m_cellCount;
    std::vector<std::uniform_real_distribution<FeatureType>> m_distribution;
    std::vector<FeatureType>                                 m_value;
};

template <typename FeatureType>
class SingleSourceGenerator
{
public:

    typedef std::shared_ptr<SingleSourceGenerator<FeatureType>> SharedPointer;

    SingleSourceGenerator():
    m_featureCount( 0 )
    {
    }

    unsigned int getFeatureCount() const
    {
        return m_featureCount;
    }

    void addFeatureGenerator( FeatureGenerator<FeatureType>::SharedPointer generator )
    {
        m_featureCount += generator->getFeatureCount();
        m_features.push_back( generator );
    }

    void generatePoint( FeatureGenerator<FeatureType>::RandomEngine & engine, Table<FeatureType> & points, unsigned int point )
    {
        // Generate data for the point.
        assert( points.getColumnCount() == m_featureCount );
        auto featureIterator = points.begin() + points.getColumnCount() * point;
        for ( auto & feature : m_features )
        {
            featureIterator = feature->generate( engine, featureIterator );
        }
    }

private:

    unsigned int                                                       m_featureCount;
    std::vector<typename FeatureGenerator<FeatureType>::SharedPointer> m_features;
};

template <typename FeatureType>
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

    unsigned int getFeatureCount() const
    {
        return m_featureCount;
    }

    virtual void generate( unsigned int pointCount, Table<FeatureType> & points, Table<Label> & labels )
    {
        // Create the tables.
        points = Table<FeatureType>( pointCount, m_featureCount );
        labels = Table<Label>( pointCount, 1 );

        // Generate the data.
        for ( unsigned int point = 0; point < pointCount; ++point )
        {
            // Select a label (ground truth).
            Label label = m_sourceDistribution( m_engine );
            assert( label < m_sources.size() );
            labels( point, 0 ) = label;

            // Let the source of that label generate the data.
            m_sources[label]->generatePoint( m_engine, points, point );
        }
    }

    void addSource( FeatureType relativeFrequency, SingleSourceGenerator<FeatureType>::SharedPointer source )
    {
        // Add the source to the list of sources.
        assert( source->getFeatureCount() == m_featureCount );
        m_sources.push_back( source );
        m_frequencies.push_back( relativeFrequency );

        // Create a distribution to select the sources.
        std::vector<FeatureType> indices( m_sources.size() + 1 );
        std::iota( indices.begin(), indices.end(), 0 );
        m_sourceDistribution = std::piecewise_constant_distribution<FeatureType>( indices.begin(), indices.end(), m_frequencies.begin() );
    }

private:

    unsigned int                                                            m_featureCount;
    FeatureGenerator<FeatureType>::RandomEngine                             m_engine;
    std::vector<typename SingleSourceGenerator<FeatureType>::SharedPointer> m_sources;
    std::vector<FeatureType>                                                m_frequencies;
    std::piecewise_constant_distribution<FeatureType>                       m_sourceDistribution;
};

/**
 * Parse a data generator from a configuration file.
 */
template <typename FeatureType>
DataGenerator<FeatureType>::SharedPointer parseDataGenerator( std::istream & in, unsigned int seed = 0 )
{
    // Parse the datasource type name.
    GenericParser parser( in );
    auto          datasourceType = parser.parseIdentifier();
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
            parser.consume( "{" );

            // Parse the feature generators.
            typename SingleSourceGenerator<FeatureType>::SharedPointer source( new SingleSourceGenerator<FeatureType>() );
            while ( source->getFeatureCount() < featureCount )
            {
                // Parse the type of the distribution.
                auto distributionType = parser.parseIdentifier();
                parser.consume( '(' );
                if ( distributionType == "uniform" )
                {
                    // Add a generator to the source.
                    auto lowerBound = parser.parseValue<double>();
                    parser.consume( ',' );
                    auto upperBound = parser.parseValue<double>();
                    source->addFeatureGenerator( typename FeatureGenerator<FeatureType>::SharedPointer( new UniformFeatureGenerator( lowerBound, upperBound ) ) );
                }
                else if ( distributionType == "gaussian" )
                {
                    // Add a generator to the source.
                    auto mean = parser.parseValue<double>();
                    parser.consume( ',' );
                    auto standardDeviation = parser.parseValue<double>();
                    source->addFeatureGenerator( typename FeatureGenerator<FeatureType>::SharedPointer( new GaussianFeatureGenerator( mean, standardDeviation ) ) );
                }
                else if ( distributionType == "annulus" )
                {
                    // Add a generator to the source.
                    auto minRadius = parser.parseValue<double>();
                    parser.consume( ',' );
                    auto maxRadius = parser.parseValue<double>();
                    source->addFeatureGenerator( typename FeatureGenerator<FeatureType>::SharedPointer( new AnnulusFeatureGenerator( minRadius, maxRadius ) ) );
                }
                else if ( distributionType == "checkerboard" )
                {
                    // Add a generator to the source.
                    auto colorName = parser.parseIdentifier();
                    auto color = CheckerboardFeatureGenerator<FeatureType>::Color::BLACK;
                    if ( colorName == "white" )
                        color = CheckerboardFeatureGenerator<FeatureType>::Color::WHITE;
                    else if ( colorName != "black" )
                        throw ParseError( "Unrecognized checkerboard color name: " + colorName );
                    parser.consume( ',' );
                    auto dimensionCount = parser.parseValue<unsigned int>();
                    typename CheckerboardFeatureGenerator<FeatureType>::SharedPointer checkerboard( new CheckerboardFeatureGenerator<FeatureType>( color ) );
                    for ( unsigned int dimension = 0; dimension < dimensionCount; ++ dimension )
                    {
                        parser.consume( ',' );
                        unsigned int cellCount = parser.parseValue<unsigned int>();
                        parser.consume( ',' );
                        FeatureType cellSize = parser.parseValue<FeatureType>();
                        checkerboard->addDimension( cellCount, cellSize );
                    }
                    source->addFeatureGenerator( checkerboard );
                }
                else
                    throw ParseError( "Unrecognized random distribution type: " + distributionType );
                parser.consume( ')' );
                parser.consume( ';' );
            }

            // Consume the closing bracket of the source definition.
            parser.consume( "}" );

            // Add the source to the multisource.
            if ( source->getFeatureCount() != featureCount )
                throw ParseError( "The feature count of the source differs from the feature count of the containing multi-source." );
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

} // namespace balsa

#endif // DATAGENERATOR_H
