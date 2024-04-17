#ifndef DATATOOLS_H
#define DATATOOLS_H

#include <algorithm>
#include <sstream>
#include <valarray>
#include <vector>

#include "datatypes.h"

namespace balsa
{

/**
 * A table for counting the number of occurrences of various labels in a set of data points.
 */
class LabelFrequencyTable
{
public:

    /**
     * Constructs a frequency table.
     * \param exclusiveUpperLimit All counted values must be strictly below this limit.
     */
    LabelFrequencyTable( Label exclusiveUpperLimit ):
    m_data( Label( 0 ), static_cast<std::size_t>( exclusiveUpperLimit ) ),
    m_total( 0 )
    {
    }

    /**
     * Creates a frequency table from a list of labels.
     */
    template <typename InputIterator>
    LabelFrequencyTable( InputIterator labelStart, InputIterator labelEnd )
    {
        // Create a temporary vector of label frequency counts (since valarray's
        // resize() function initializes the entire valarray to zero).
        std::vector<std::size_t> counts;
        for ( auto label( labelStart ); label != labelEnd; ++label )
        {
            // Grow the count table if a large label is found.
            if ( *label >= counts.size() ) counts.resize( *label + 1 );

            // Count the label.
            ++counts[*label];
        }

        // Copy the computed frequency counts to the valarray member variable.
        m_data.resize( counts.size() );
        std::copy( counts.begin(), counts.end(), std::begin( m_data ) );

        // Calculate the total size.
        m_total = m_data.sum();
    }

    /**
     * Increment the count of a label by 1.
     * \param value The label of a data point.
     * \pre label < exclusiveUpperLimit (constructor parameter).
     */
    void increment( Label label )
    {
        assert( label < m_data.size() );
        ++m_data[label];
        ++m_total;
    }

    /**
     * Subtract 1 from the count of a label.
     * \pre getCount( label ) > 0
     */
    void decrement( Label label )
    {
        assert( m_data[label] > 0 );
        --m_data[label];
        --m_total;
    }

    /**
     * Returns the stored count of a particular label.
     */
    std::size_t getCount( Label label ) const
    {
        assert( label < m_data.size() );
        return m_data[label];
    }

    /**
     * Returns the total of all counts.
     */
    std::size_t getTotal() const
    {
        return m_total;
    }

    /**
     * Returns the number of distinct, consecutive label values that can be counted in this table.
     */
    std::size_t size() const
    {
        return m_data.size();
    }

    /**
     * Calculate the Gini impurity of the dataset, based on the stored label counts.
     * \pre The table may not be empty.
     */
    template <typename FloatType>
    FloatType giniImpurity() const
    {
        assert( m_total > 0 );
        auto squaredCounts = m_data * m_data;
        return FloatType( 1.0 ) - static_cast<FloatType>( squaredCounts.sum() ) / ( m_total * m_total );
    }

    /**
     * Returns the lowest label with the highest count.
     */
    Label getMostFrequentLabel() const
    {
        // Find the lowest label with the highest count.
        std::size_t bestCount = 0;
        Label       best      = 0;
        for ( Label l = 0; l < m_data.size(); ++l )
        {
            if ( m_data[l] <= bestCount ) continue;
            best      = l;
            bestCount = m_data[l];
        }
        return best;
    }

    /**
     * Class invariant. N.B. this is expensive to compute.
     */
    bool invariant()
    {
        return m_data.sum() == m_total;
    }

    /**
     * Return a textual representation for debugging purposes.
     */
    const std::string asText() const
    {
        std::stringstream ss;
        if ( m_data.size() == 0 ) return "(No entries)";
        ss << m_data[0];
        for ( Label l = 1; l < m_data.size(); ++l ) ss << " " << static_cast<unsigned int>( m_data[l] );
        return ss.str();
    }

private:

    std::valarray<std::size_t> m_data;
    std::size_t                m_total;
};

/**
 * An axis-aligned division between two sets of points in a multidimensional
 * feature-space. N.B. the split feature value is an exclusive upper bound.
 */
template <typename FeatureType>
class Split
{
public:

    Split( FeatureID feature = 0, FeatureType value = 0 ):
    m_feature( feature ),
    m_value( value )
    {
    }

    FeatureID getFeatureID() const
    {
        return m_feature;
    }

    FeatureType getFeatureValue() const
    {
        return m_value;
    }

private:

    FeatureID   m_feature;
    FeatureType m_value;
};

} // namespace balsa

#endif // DATATOOLS_H
