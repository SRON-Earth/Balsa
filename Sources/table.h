#ifndef TABLE_H
#define TABLE_H

#include <cassert>
#include <fstream>
#include <iomanip>
#include <vector>

#include "exceptions.h"
#include "serdes.h"

/**
 * A row-major MxN data matrix that can be loaded and stored efficiently.
 * N.B. the Table does not support linear algebra operations.
 * \tparam CellType The data typ of each (x,y) entry.
 */
template <typename CellType>
class Table
{

public:

    typedef std::vector<CellType>::iterator       Iterator;
    typedef std::vector<CellType>::const_iterator ConstIterator;

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
     * \param rowNumber
     */
    std::size_t getColumnOfRowMaximum( std::size_t rowNumber ) const
    {
        auto rowData    = m_data.begin() + rowNumber * m_columnCount;
        auto rowDataEnd = rowData + m_columnCount;
        auto largest    = std::max_element( rowData, rowDataEnd );
        return std::distance( rowData, largest );
    }

    /**
     * Read-only access a element by row and column.
     */
    const CellType & operator()( std::size_t row, std::size_t column ) const
    {
        return m_data[row * m_columnCount + column];
    }

    /**
     * Read-write access an element by row and column.
     */
    CellType & operator()( std::size_t row, std::size_t column )
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
        return m_data.size() / m_columnCount;
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
     * Read cell data into the table from a binary stream.
     */
    void readCellData( std::istream & binIn )
    {
        // Read the raw binary data from the stream.
        if ( !binIn.good() ) throw ParseError( "The stream is not readable." );
        binIn.read( reinterpret_cast<char *>( m_data.data() ), m_data.size() * sizeof( CellType ) );
    }

    /**
     * Read the cell data from a stream and convert it on the fly.
     */
    template <typename SourceType>
    void readCellDataAs( std::istream & binIn )
    {
        for ( auto it( m_data.begin() ), end( m_data.end() ); it != end; ++it )
        {
            *it = deserialize<SourceType>( binIn );
        }
        if ( binIn.fail() ) throw ParseError( "Read failed." );
    }

    /**
     * Read a Table<CellType> from a file with a potentially different cell type, and convert cells on the fly.
     */
    static Table<CellType> readFileAs( const std::string & filename )
    {
        // Read the type and geometry of the table in the file.
        std::ifstream binIn;
        binIn.open( filename, std::ios::binary );
        std::size_t rows, cols;
        std::string sourceType;
        parseTableSpecification( binIn, rows, cols, sourceType );

        // Allocate a result table
        Table<CellType> result( rows, cols );

        // Read the table, convert if necessary.
        expect( binIn, "data", "Missing data marker." );
        auto destinationType = getTypeName<CellType>();
        if ( destinationType == sourceType )
        {
            // No conversion is necessary if source and destination types are the same.
            result.readCellData( binIn );
        }
        else if ( sourceType == getTypeName<float>() )
        {
            // Read as floats, convert to target type.
            result.readCellDataAs<float>( binIn );
        }
        else if ( sourceType == getTypeName<int32_t>() )
        {
            // Read as floats, convert to target type.
            result.readCellDataAs<int32_t>( binIn );
        }
        else if ( sourceType == getTypeName<uint8_t>() )
        {
            // Read as floats, convert to target type.
            result.readCellDataAs<uint8_t>( binIn );
        }
        else
        {
            throw ParseError( "Unsupported conversion from " + sourceType + " to " + destinationType );
        }

        return result;
    }

    /**
     * Serialize the table to a binary output stream.
     */
    void serialize( std::ostream & binOut ) const
    {
        // Write the table header.
        binOut.write( "tabl", 4 );
        auto typeName = getTypeName<CellType>();
        binOut.write( typeName.c_str(), typeName.size() );

        // Write the dimensions.
        binOut.write( "rows", 4 );
        ::serialize( binOut, static_cast<uint32_t>( getRowCount() ) );
        binOut.write( "cols", 4 );
        ::serialize( binOut, static_cast<uint32_t>( getColumnCount() ) );

        // Write the data
        binOut.write( "data", 4 );
        binOut.write( reinterpret_cast<const char *>( &*m_data.begin() ), m_data.size() * sizeof( CellType ) );
    }

    /**
     * Parses the cell type name, row count, and column count from a binary stream.
     * The stream is consumed until the start of the 'data' block. The 'data' label is not consumed.
     */
    static void parseTableSpecification( std::istream & binIn, std::size_t & rowCount, std::size_t & columnCount, std::string & typeName )
    {
        // Parse the table marker.
        expect( binIn, "tabl", "Missing or malformed table header." );

        // Parse the type name.
        typeName = getFixedSizeToken( binIn, 4 );

        // Parse the table dimensions.
        expect( binIn, "rows", "Missing rows marker." );
        rowCount = deserialize<std::uint32_t>( binIn );
        expect( binIn, "cols", "Missing cols marker." );
        columnCount = deserialize<std::uint32_t>( binIn );
    }

private:

    // Returns true iff the internal datastructure is consistent.
    bool invariant() const
    {
        return ( m_data.size() % m_columnCount ) == 0;
    }

    std::size_t           m_columnCount;
    std::vector<CellType> m_data;
};

/**
 * Reads a Table from a binary input stream.
 * \pre The stored table must be of the same cell type.
 */
template <typename CellType>
std::istream & operator>>( std::istream & binIn, Table<CellType> & table )
{
    // Read the type and geometry of the table.
    std::size_t rows, cols;
    std::string typeName;
    Table<CellType>::parseTableSpecification( binIn, rows, cols, typeName );

    // Check the type name.
    if ( typeName != getTypeName<CellType>() ) throw ParseError( "Source/destination type mismatch." );

    // Allocate a table and parse the data.
    expect( binIn, "data", "Missing data marker." );
    table = Table<CellType>( rows, cols );
    table.readCellData( binIn );

    // Return the stream.
    return binIn;
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
std::ostream & operator<<( std::ostream & out, const Table<uint8_t> & table )
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

#endif // TABLE_H
