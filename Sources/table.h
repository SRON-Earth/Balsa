#ifndef TABLE_H
#define TABLE_H

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iomanip>
#include <vector>

#include "genericparser.h"
#include "exceptions.h"
#include "serdes.h"

namespace balsa
{

/**
 * A row-major MxN data matrix that can be loaded and stored efficiently.
 * N.B. the Table does not support linear algebra operations.
 * \tparam CellType The data type of each (x,y) entry.
 */
template <typename CellType>
class Table
{

public:

    typedef typename std::vector<CellType>::iterator        Iterator;
    typedef typename std::vector<CellType>::const_iterator  ConstIterator;
    typedef typename std::vector<CellType>::reference       Reference;
    typedef typename std::vector<CellType>::const_reference ConstReference;

    Table():
    m_columnCount( 0 )
    {
    }

    /**
     * Constructs an empty table with the specified number of columns.
     * \param columnCount The number of columns in the table.
     */
    Table( std::size_t columnCount ):
    m_columnCount( columnCount )
    {
    }

    /**
     * Constructs a table with the specified number of rows and columns, with each cell initialized to the specified value.
     * \param columnCount The number of columns in the table.
     */
    Table( std::size_t rowCount, std::size_t columnCount, CellType initialValue = CellType( 0 ) ):
    m_columnCount( columnCount )
    {
        m_data.resize( rowCount * columnCount, initialValue );
    }

    /**
     * Find the largest element in a row and return its column number.
     * In case of a tie, the lowest tied column number is returned.
     * \param rowNumber The row that is to be scanned.
     */
    std::size_t getColumnOfRowMaximum( std::size_t rowNumber ) const
    {
        auto rowData    = m_data.begin() + rowNumber * m_columnCount;
        auto rowDataEnd = rowData + m_columnCount;
        auto largest    = std::max_element( rowData, rowDataEnd );
        return std::distance( rowData, largest );
    }

    /**
     * Find the largest element in a row and return its column number, after applying a weight.
     * In case of a tie, the lowest tied column number is returned.
     * \param rowNumber The row that is to be scanned.
     * \param weights The list of non-negative weight factors that will be applied to the counts.
     */
    std::size_t getColumnOfWeightedRowMaximum( std::size_t rowNumber, const std::vector<float> &weights ) const
    {
        // Find the maximum of the weighted values.
        auto rowData    =  m_data.begin() + rowNumber * m_columnCount;
        double topScore = 0;
        std::size_t topColumn = 0;
        for ( std::size_t column = 0; column < m_columnCount; ++column )
        {
            float score = rowData[column] * weights[column];
            if ( score <= topScore ) continue;
            topColumn = column;
            topScore  = score;
        }
        return topColumn;
    }

    /**
     * Read-only access a element by row and column.
     */
    ConstReference operator()( std::size_t row, std::size_t column ) const
    {
        return m_data[row * m_columnCount + column];
    }

    /**
     * Read-write access an element by row and column.
     */
    Reference operator()( std::size_t row, std::size_t column )
    {
        return m_data[row * m_columnCount + column];
    }

    /**
     * Returns an iterator that traverses all cells in row-major order.
     */
    ConstIterator begin() const
    {
        return m_data.begin();
    }

    /**
     * Returns an iterator that points ot the end of the data.
     */
    ConstIterator end() const
    {
        return m_data.end();
    }

    /**
     * Returns an iterator that traverses all cells in row-major order.
     */
    Iterator begin()
    {
        return m_data.begin();
    }

    /**
     * Returns an iterator that points ot the end of the data.
     */
    Iterator end()
    {
        return m_data.end();
    }

    /**
     * Append rows to the table.
     * \pre The total number of elements between rowStart and rowEnd must be a
     * multiple of the row length.
     */
    template <typename InputIterator>
    void append( InputIterator rowStart, InputIterator rowEnd )
    {
        // Copy the rows and check consistency.
        std::copy( rowStart, rowEnd, std::back_inserter( m_data ) );
        assert( invariant() );
    }

    /**
     * Reserve space for a number of rows.
     */
    void reserveRows( std::size_t rowCount )
    {
        m_data.reserve( rowCount * m_columnCount );
    }

    /**
     * Returns the number of columns.
     */
    std::size_t getColumnCount() const
    {
        return m_columnCount;
    }

    /**
     * Returns the number of rows.
     */
    std::size_t getRowCount() const
    {
        return ( m_columnCount == 0 ) ? 0 : ( m_data.size() / m_columnCount );
    }

    /**
     * Add the cells of another table to this table element-wise.
     * \pre Dimensions must match.
     */
    Table<CellType> & operator+=( const Table<CellType> & other )
    {
        // Check preconditions.
        assert( other.m_columnCount == m_columnCount );
        assert( other.m_data.size() == m_data.size() );

        // Add the data element-wise.
        auto it1( m_data.begin() ), end1( m_data.end() ); // Non-const.
        auto it2( other.m_data.begin() );                 // Const.
        for ( ; it1 != end1; ++it1, ++it2 )
        {
            *it1 += *it2;
        }

        return *this;
    }

    /**
     * Test if this and another table are equal, that is, have the same shape
     * and contain the same values.
     */
    bool operator==( const Table<CellType> & other ) const
    {
        // Check preconditions.
        if ( other.m_columnCount != m_columnCount )
            return false;
        if ( other.m_data.size() != m_data.size() )
            return false;
        return std::equal( m_data.begin(), m_data.end(), other.m_data.begin() );
    }

    /**
     * Test if this and another table are different, that is, have a different
     * shape or contain different values.
     */
    bool operator!=( const Table<CellType> & other ) const
    {
        return !( *this == other );
    }

    /**
     * Read cell data into the table from a binary stream.
     */
    void readCellData( std::istream & stream )
    {
        // Read the raw binary data from the stream.
        if ( !stream.good() ) throw ParseError( "The stream is not readable." );
        stream.read( reinterpret_cast<char *>( m_data.data() ), m_data.size() * sizeof( CellType ) );
    }

    /**
     * Read the cell data from a stream and convert it on the fly.
     */
    template <typename SourceType>
    void readCellDataAs( std::istream & stream )
    {
        for ( auto it( m_data.begin() ), end( m_data.end() ); it != end; ++it )
        {
            *it = balsa::deserialize<SourceType>( stream );
        }
        if ( stream.fail() ) throw ParseError( "Read failed." );
    }

    /**
     * Write cell data from the table into a binary stream.
     */
    void writeCellData( std::ostream & stream ) const
    {
        // Read the raw binary data from the stream.
        if ( !stream.good() ) throw ParseError( "The stream is not writable." );
        stream.write( reinterpret_cast<const char *>( m_data.data() ), m_data.size() * sizeof( CellType ) );
    }

private:

    // Returns true iff the internal datastructure is consistent.
    bool invariant() const
    {
        return ( m_columnCount == 0 ) ? ( m_data.size() == 0 ) : ( ( m_data.size() % m_columnCount ) == 0 );
    }

    std::size_t           m_columnCount;
    std::vector<CellType> m_data;
};

/**
 * Specialization for bool tables.
 *
 * Conversion is always necessary for type bool because std::vector<bool> can
 * and does use space saving storage for booleans.
 */
template <>
inline void Table<bool>::readCellData( std::istream & stream )
{
    readCellDataAs<bool>( stream );
}

/**
 * Writes table contents to a text stream in human-readable form.
 */
template <typename CellType>
std::ostream & operator<<( std::ostream & out, const Table<CellType> & table )
{
    // Write the cell data and row numbers.
    for ( unsigned int row = 0; row < table.getRowCount(); ++row )
    {
        out << std::setw( 4 ) << std::left << row << ':';
        for ( unsigned int col = 0; col < table.getColumnCount(); ++col ) out << ' ' << std::setw( 8 ) << std::left << table( row, col );
        out << std::endl;
    }

    return out;
}

/**
 * Specialization for uint8_t tables.
 */
inline std::ostream & operator<<( std::ostream & out, const Table<uint8_t> & table )
{
    // Write the cell data and row numbers.
    for ( unsigned int row = 0; row < table.getRowCount(); ++row )
    {
        out << std::setw( 4 ) << std::left << row << ':';
        for ( unsigned int col = 0; col < table.getColumnCount(); ++col ) out << ' ' << std::setw( 4 ) << std::left << static_cast<unsigned int>( table( row, col ) );
        out << std::endl;
    }

    return out;
}

template<typename CellType>
Table<CellType> parseCSV( std::istream &in )
{
    // Create a generic parser that does not consume newlines as whitespace.
    GenericParser parser( in, " \t\r" );

    // Consume whitespace and empty lines.
    while ( !parser.atEOF() )
    {
        parser.consumeWhitespace();
        if ( parser.peek() != '\n' ) break;
        parser.consume( '\n' );
    }

    // Parse the first row with data. This determines the width of the table.
    std::vector<CellType> firstRow;
    while ( !parser.atEOF() && parser.peek() != '\n' )
    {
        // Parse a value.
        parser.consumeWhitespace();
        firstRow.push_back( parser.parseValue<CellType>() );
        parser.consumeWhitespace();

        // Break on line endings.
        if ( parser.peek() == '\n' )
        {
            parser.consume( '\n' );
            break;
        }

        // Consume the separator.
        parser.consume( ',' );
    }

    if ( firstRow.size() == 0 ) throw ParseError( "No data in CSV file." );

    // Create an empty table.
    Table<CellType> result( firstRow.size() );
    result.append( firstRow.begin(), firstRow.end() );

    // Parse all remaining rows.
    while ( !parser.atEOF() )
    {
        // Consume whitespace and empty lines.
        while ( !parser.atEOF() )
        {
            parser.consumeWhitespace();
            if ( parser.peek() != '\n' ) break;
            parser.consume( '\n' );
        }

        // Parse an actual row.
        std::vector<CellType> row;
        while ( !parser.atEOF() )
        {
            // Parse a value.
            parser.consumeWhitespace();
            row.push_back( parser.parseValue<CellType>() );
            parser.consumeWhitespace();

            // Break on line endings.
            if ( parser.peek() == '\n' )
            {
                parser.consume( '\n' );
                break;
            }

            // Check for a valid separator.
            parser.consume( ',' );
        }

        // Add the row to the table.
        if ( row.size() == 0 ) continue;
        if ( row.size() != firstRow.size() ) throw ParseError( "CSV rows must be of equal length." );
        result.append( row.begin(), row.end() );
    }

    return result;
}

} // namespace balsa

#endif // TABLE_H
