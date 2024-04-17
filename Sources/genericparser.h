#ifndef GENERICPARSER_H
#define GENERICPARSER_H

#include <iostream>
#include <string>

#include "exceptions.h"
#include "serdes.h"

namespace balsa
{

/**
 * A non application specific parser for processing text streams.
 */
class GenericParser
{
public:

    GenericParser( std::istream & in ):
    m_whitespace( " \t\r\n" ),
    m_in( in )
    {
        assert( in.good() );
    }

    /**
     * Removes whitespace and the specified character.
     * Throws an exception if the first non-whitespace character is not the expected literal.
     */
    void consume( char literal )
    {
        consumeWhitespace();
        char c = m_in.get();
        if ( m_in.fail() ) throw ParseError( "Could not read from stream." );
        if ( c != literal ) throw ParseError( std::string( "Expected '" ) + literal + "', got '" + c + "'." );
    }

    /**
     * Consume whitespace followed by the specified literal value.
     */
    void consume( const std::string & literal )
    {
        consumeWhitespace();
        auto token = getFixedSizeToken( m_in, literal.size() );
        if ( token != literal ) throw ParseError( "Expected literal '" + literal + "', got '" + token + "'." );
    }

    /**
     * Parse an identifier consisting of a letter followed by zero or more letters, numbers and underscores.
     */
    std::string parseIdentifier()
    {
        // Consume whitespace and parse the first character.
        consumeWhitespace();
        std::stringstream ss;
        if ( !isalpha( m_in.peek() ) ) throw ParseError( "Expected an identifier." );
        ss << static_cast<char>( m_in.get() );

        // Parse the remaining characters.
        while ( isalpha( m_in.peek() ) || isdigit( m_in.peek() ) ) ss << static_cast<char>( m_in.get() );
        return ss.str();
    }

    /**
     * Consumes leading whitespace.
     */
    void consumeWhitespace()
    {
        while ( m_whitespace.contains( m_in.peek() ) )
        {
            m_in.get();
            if ( m_in.fail() ) throw ParseError( "Error reading from stream." );
        }
    }

    template <typename T>
    T parseValue()
    {
        T result;
        m_in >> result;
        if ( !m_in ) throw ParseError( "Could not convert the input to a value of the expected type." );
        return result;
    }

    char peek() const
    {
        return m_in.peek();
    }

private:

    std::string    m_whitespace;
    std::istream & m_in;
};

} // namespace balsa

#endif // GENERICPARSER_H
