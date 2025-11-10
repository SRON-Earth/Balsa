#ifndef WEIGHTEDCOIN_H
#define WEIGHTEDCOIN_H

#include <cassert>
#include <mutex>
#include <random>

namespace balsa
{

/**
 * A thread safe random number generator.
 */
template <typename T_RNG = std::mt19937>
class ThreadSafeRandomNumberGenerator
{
public:

    typedef typename T_RNG::result_type ValueType;

    /**
     * Constructor.
     */
    ThreadSafeRandomNumberGenerator():
    m_rng( std::random_device{}() )
    {
    }

    /**
     * Seed the random number generator.
     */
    void seed( ValueType value )
    {
        std::lock_guard lock( m_mutex );
        m_rng.seed( value );
    }

    /**
     * Generate a random number.
     */
    ValueType next()
    {
        std::lock_guard lock( m_mutex );
        return m_rng();
    }

private:

    T_RNG      m_rng;
    std::mutex m_mutex;
};

/**
 * Thread safe random number generator type used for seeding thread local random
 * generators.
 */
typedef ThreadSafeRandomNumberGenerator<> MasterSeedSequence;

/**
 * Return a reference to a singleton thread safe random number generator that
 * can be used for seeding thread local random number generators.
 */
MasterSeedSequence & getMasterSeedSequence();

/**
 * Coin that can be flipped with a specific probability of being true.
 */
template <typename T_RNG = std::mt19937>
class WeightedCoin
{
public:

    typedef typename T_RNG::result_type ValueType;

    /**
     * Constructor.
     */
    WeightedCoin():
    m_rng( std::random_device{}() )
    {
    }

    /**
     * Seed the random number generator used for flipping the coin.
     */
    void seed( ValueType value )
    {
        m_rng.seed( value );
    }

    /**
     * Returns a random boolean, with the probability of it being true equal to
     * an integer fraction.
     */
    bool flip( unsigned int numerator, unsigned int denominator )
    {
        assert( numerator <= denominator );
        if ( numerator == denominator ) return true;
        std::uniform_int_distribution<unsigned int> dist( 1, denominator );
        return dist( m_rng ) <= numerator;
    }

private:

    T_RNG m_rng;
};

} // namespace balsa

#endif // WEIGHTEDCOIN_H
