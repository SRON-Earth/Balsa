#include "utilities.h"

bool randomBool( unsigned int numerator, unsigned int denominator )
{
    assert( numerator <=  denominator );
    if ( numerator == denominator ) return true;

    static std::random_device dev;
    static std::mt19937 rng( dev() );
    std::uniform_int_distribution<std::mt19937::result_type> dist( 1, denominator );
    return dist(rng) <= numerator;
}
