#ifndef SERDES_H
#define SERDES_H

#include <algorithm>
#include <cstdint>
#include <istream>
#include <ostream>
#include <type_traits>

#include "exceptions.h"

/**
 * Serialize a POD (plain old data) value to a binary output stream.
 */
template <typename T>
void serialize( std::ostream & os, const T & value )
{
    static_assert( std::is_standard_layout<T>::value && std::is_trivial<T>::value, "Generic serialization is implemented for POD types only." );
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
    static_assert( std::is_standard_layout<T>::value && std::is_trivial<T>::value, "Generic deserialization is implemented for POD types only." );
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

/**
 * Read a fixed-size token from a stream.
 */
inline std::string getFixedSizeToken( std::istream & is, std::size_t size )
{
    std::string token;
    auto        it = std::istreambuf_iterator<char>( is );
    std::copy_n( it, size, std::back_inserter( token ) );
    ++it;
    if ( is.fail() ) throw ParseError( "Read failed." );
    return token;
}

/**
 * Peek at a fixed-size token.
 */
inline std::string peekFixedSizeToken( std::istream & is, std::size_t size )
{
    auto        position = is.tellg();
    std::string token    = getFixedSizeToken( is, size );
    is.seekg( position );
    return token;
}

/**
 * Read an expected sequence of characters from a stream, throw an exception if the is a mismatch.
 */
inline void expect( std::istream & is, const std::string & sequence, const std::string & errorMessage )
{
    std::string token = getFixedSizeToken( is, sequence.size() );
    if ( token != sequence ) throw ParseError( errorMessage );
}

/**
 * Read until a separator is encountered. Separators are not consumed.
 */
std::string getNextToken( std::istream & is, const std::string & separators )
{
    std::stringstream token;
    while ( !separators.contains( is.peek() ) ) token << ( is.get() );
    return token.str();
}

template <typename Type>
std::string getTypeName()
{
    static_assert( sizeof( Type ) != sizeof( Type ), "Unsupported type." );
    return "";
}

template <>
std::string getTypeName<bool>()
{
    return "bool";
}

template <>
std::string getTypeName<float>()
{
    return "fl32";
}

template <>
std::string getTypeName<double>()
{
    return "fl64";
}

template <>
std::string getTypeName<uint32_t>()
{
    return "ui32";
}

template <>
std::string getTypeName<int32_t>()
{
    return "in32";
}

template <>
std::string getTypeName<int16_t>()
{
    return "in16";
}

template <>
std::string getTypeName<uint16_t>()
{
    return "ui16";
}

template <>
std::string getTypeName<uint8_t>()
{
    return "ui08";
}

#endif // SERDES_H
