#ifndef WEIGHTEDCOIN_H
#define WEIGHTEDCOIN_H

#include <random>
#include <mutex>

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
  ThreadSafeRandomNumberGenerator()
  : m_rng( std::random_device{}() )
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
class WeightedCoin
{
public:

  /**
   * Constructor.
   */
  WeightedCoin();

  /**
   * Returns a random boolean, with probability of being true equal to an integer fraction.
   */
  bool flip( unsigned int numerator, unsigned int denominator );

private:

  std::mt19937 m_rng;

};

#endif // WEIGHTEDCOIN_H
