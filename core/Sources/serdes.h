#ifndef SERDES_H
#define SERDES_H

#include <istream>
#include <ostream>
#include <string>
#include <type_traits>

namespace balsa
{

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
void serialize( std::ostream & os, const bool & value );

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
bool deserialize( std::istream & is );

/**
 * Read a fixed-size token from a stream.
 */
std::string getFixedSizeToken( std::istream & is, std::size_t size );

/**
 * Peek at a fixed-size token.
 */
std::string peekFixedSizeToken( std::istream & is, std::size_t size );

/**
 * Read an expected sequence of characters from a stream, throw an exception if the is a mismatch.
 */
void expect( std::istream & is, const std::string & sequence, const std::string & errorMessage );

/**
 * Read until a separator is encountered. Separators are not consumed.
 */
std::string getNextToken( std::istream & is, const std::string & separators );

} // namespace balsa

#endif // SERDES_H
