#ifndef UTILITIES_H
#define UTILITIES_H

#include <algorithm>
#include <iostream>
#include <random>
#include <tuple>
#include <vector>

#include "datarepresentation.h"

/**
 * An index for traversing points in a dataset in order of each feature.
 */
class FeatureIndex
{
public:

    typedef std::tuple<double, bool, DataPointID> Entry;
    typedef std::vector<Entry> SingleFeatureIndex;

    FeatureIndex( const TrainingDataSet & dataset )
    : m_trueCount( 0 )
    {
        // Create a sorted index for each feature.
        m_featureIndices.clear();
        m_featureIndices.reserve( dataset.getFeatureCount() );

        for ( unsigned int feature = 0; feature < dataset.getFeatureCount(); ++feature )
        {
            // Create the index for this feature, and give it enough capacity.
            m_featureIndices.push_back( std::vector<Entry>() );
            auto & index = *m_featureIndices.rbegin();
            index.reserve( dataset.size() );

            // Create entries for each point in the dataset, and count the 'true' labels.
            m_trueCount = 0;
            for ( DataPointID pointID( 0 ), end( dataset.size() ); pointID < end; ++pointID )
            {
                bool label = dataset.getLabel( pointID );
                if ( label ) ++m_trueCount;
                index.push_back( Entry( dataset.getFeatureValue( pointID, feature ), label, pointID ) );
            }

            // Sort the index by feature value (the rest of the fields don't matter).
            std::sort( index.begin(), index.end() );
        }
    }

    SingleFeatureIndex::const_iterator featureBegin( unsigned int featureID ) const
    {
        return m_featureIndices[featureID].begin();
    }

    SingleFeatureIndex::const_iterator featureEnd( unsigned int featureID ) const
    {
        return m_featureIndices[featureID].end();
    }

    /**
     * Returns the number of features.
     */
    unsigned int getFeatureCount() const
    {
        return m_featureIndices.size();
    }

    /**
     * Returns the number of indexed points.
     */
    std::size_t size() const
    {
        return m_featureIndices[0].size();
    }

    /**
     * Returns the number of points labeled 'true'.
     */
    unsigned int getTrueCount() const
    {
        return m_trueCount;
    }

private:

    unsigned int m_trueCount;
    std::vector<SingleFeatureIndex> m_featureIndices;
};

/**
 * Compute the Gini impurity of a set of totalCount points, where trueCount points are labeled 'true', and the rest is
 * false.
 */
inline double giniImpurity( unsigned int trueCount, unsigned int totalCount )
{
    double t = trueCount;
    auto T   = totalCount;
    return ( 2 * t * ( 1.0 - ( t / T ) ) ) / T;
}

#endif // UTILITIES_H
