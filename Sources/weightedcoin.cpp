#include <cassert>

#include "weightedcoin.h"

MasterSeedSequence & getMasterSeedSequence()
{
	static MasterSeedSequence seedSequence;
	return seedSequence;
}

WeightedCoin::WeightedCoin()
: m_rng( getMasterSeedSequence().next() )
{
}

bool WeightedCoin::flip( unsigned int numerator, unsigned int denominator )
{
    assert( numerator <=  denominator );
    if ( numerator == denominator ) return true;
    std::uniform_int_distribution<unsigned int> dist( 1, denominator );
    return dist( m_rng ) <= numerator;
}
