#ifndef SERDES_H
#define SERDES_H

#include <cstdint>
#include <istream>
#include <ostream>
#include <type_traits>

/**
 * Serialize a POD (plain old data) value to a binary output stream.
 */
template <typename T>
void serialize( std::ostream & os, const T & value )
{
    static_assert( std::is_standard_layout<T>::value && std::is_trivial<T>::value,
        "Generic serialization is implemented for POD types only." );
    os.write( reinterpret_cast<const char *>( &value ), sizeof( T ) );
}

/**
 * Specialization that serializes values of type bool as 8-bit unsigned
 * integers.
 */
template <>
inline void serialize( std::ostream & os, const bool & value )
{
    serialize<std::uint8_t>( os, value );
}

/**
 * Deserialize a POD (plain old data) value from a binary input stream.
 */
template <typename T>
T deserialize( std::istream & is )
{
    static_assert( std::is_standard_layout<T>::value && std::is_trivial<T>::value,
        "Generic deserialization is implemented for POD types only." );
    T result;
    is.read( reinterpret_cast<char *>( &result ), sizeof( T ) );
    return result;
}

/**
 * Specialization that deserializes values of type bool as 8-bit unsigned
 * integers.
 */
template <>
inline bool deserialize( std::istream & is )
{
    return ( deserialize<std::uint8_t>( is ) != 0 );
}

#endif // SERDES_H
