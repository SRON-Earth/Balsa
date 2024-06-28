#include <algorithm>
#include <cstdint>
#include <sstream>

#include "exceptions.h"
#include "serdes.h"

namespace balsa
{

template <>
void serialize( std::ostream & os, const bool & value )
{
    serialize<uint8_t>( os, value );
}

template <>
bool deserialize( std::istream & is )
{
    return ( deserialize<uint8_t>( is ) != 0 );
}

std::string getFixedSizeToken( std::istream & is, std::size_t size )
{
    std::string token;
    auto        it = std::istreambuf_iterator<char>( is );
    std::copy_n( it, size, std::back_inserter( token ) );
    ++it;
    return token;
}

std::string peekFixedSizeToken( std::istream & is, std::size_t size )
{
    auto        position = is.tellg();
    std::string token    = getFixedSizeToken( is, size );
    is.seekg( position );
    return token;
}

void expect( std::istream & is, const std::string & sequence, const std::string & errorMessage )
{
    std::string token = getFixedSizeToken( is, sequence.size() );
    if ( token != sequence ) throw ParseError( errorMessage );
}

std::string getNextToken( std::istream & is, const std::string & separators )
{
    std::stringstream token;
    while ( separators.find( is.peek() ) != std::string::npos )
        token << ( is.get() );
    return token.str();
}

} // namespace balsa
