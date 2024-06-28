#ifndef FILEIO_H
#define FILEIO_H

#include <fstream>
#include <optional>
#include <string>

#include "datatypes.h"
#include "decisiontreeclassifier.h"
#include "exceptions.h"
#include "table.h"

namespace balsa
{

enum class ScalarTypeID
{
    UINT8,
    UINT16,
    UINT32,
    INT8,
    INT16,
    INT32,
    FLOAT,
    DOUBLE,
    BOOL
};

template <typename Type>
ScalarTypeID getScalarTypeID()
{
    static_assert( sizeof( Type ) != sizeof( Type ), "Unsupported feature type." );
    return static_cast<ScalarTypeID>( 0 );
}

enum class FeatureTypeID
{
    FLOAT,
    DOUBLE
};

template <typename Type>
FeatureTypeID getFeatureTypeID()
{
    static_assert( sizeof( Type ) != sizeof( Type ), "Unsupported feature type." );
    return static_cast<FeatureTypeID>( 0 );
}

class ForestHeader
{
public:

    unsigned char classCount;
    unsigned char featureCount;
    FeatureTypeID featureTypeID;
};

class TreeHeader
{
public:

    unsigned char classCount;
    unsigned char featureCount;
    FeatureTypeID featureTypeID;
};

class TableHeader
{
public:

    unsigned int rowCount;
    unsigned int columnCount;
    ScalarTypeID scalarTypeID;
};

template <typename FeatureType>
class TreeData
{
public:

    unsigned int       classCount;
    unsigned int       featureCount;
    Table<NodeID>      leftChildID;
    Table<NodeID>      rightChildID;
    Table<FeatureID>   splitFeatureID;
    Table<FeatureType> splitValue;
    Table<Label>       label;
};

class BalsaFileParser
{
public:

    BalsaFileParser( const std::string & filename );

    unsigned int getFileMajorVersion() const;
    unsigned int getFileMinorVersion() const;

    std::optional<std::string>  getCreatorName() const;
    std::optional<unsigned int> getCreatorMajorVersion() const;
    std::optional<unsigned int> getCreatorMinorVersion() const;
    std::optional<unsigned int> getCreatorPatchVersion() const;

    bool atEOF();

    bool atForest();

    bool atEndOfForest();

    bool atTree();

    bool atTable();

    template <typename FeatureType>
    bool atTreeOfType()
    {
        return atTreeOfType( getFeatureTypeID<FeatureType>() );
    }

    template <typename CellType>
    bool atTableOfType()
    {
        return atTableOfType( getScalarTypeID<CellType>() );
    }

    ForestHeader enterForest();

    void leaveForest();

    void reenterForest();

    template <typename FeatureType>
    TreeData<FeatureType> parseTreeData()
    {
        // Parse the tree start marker.
        parseTreeStartMarker();

        // Parse the header.
        TreeHeader header = parseTreeHeader();

        // Check the feature type.
        if ( header.featureTypeID != getFeatureTypeID<FeatureType>() )
            throw ParseError( "Tree has incompatible feature type." );

        // Deserialize the rest of the tree.
        TreeData<FeatureType> result;
        result.classCount     = header.classCount;
        result.featureCount   = header.featureCount;
        result.leftChildID    = parseTable<NodeID>();
        result.rightChildID   = parseTable<NodeID>();
        result.splitFeatureID = parseTable<FeatureID>();
        result.splitValue     = parseTable<FeatureType>();
        result.label          = parseTable<Label>();

        // Parse the tree end marker.
        parseTreeEndMarker();

        // Return the result.
        return result;
    }

    template <typename FeatureIterator, typename OutputIterator>
    typename DecisionTreeClassifier<FeatureIterator, OutputIterator>::SharedPointer parseTree()
    {
        // Define the type of the classifier to parse.
        typedef DecisionTreeClassifier<FeatureIterator, OutputIterator> ClassifierType;

        // Parse the internal tree data structures.
        typedef typename ClassifierType::FeatureType FeatureType;
        TreeData<FeatureType>                        data = parseTreeData<FeatureType>();

        // Create an empty classifier.
        typename ClassifierType::SharedPointer result( new ClassifierType( data.classCount, data.featureCount ) );

        // Move assign the internal tables.
        result->m_leftChildID    = std::move( data.leftChildID );
        result->m_rightChildID   = std::move( data.rightChildID );
        result->m_splitFeatureID = std::move( data.splitFeatureID );
        result->m_splitValue     = std::move( data.splitValue );
        result->m_label          = std::move( data.label );

        // Return the result.
        return result;
    }

    template <typename CellType>
    Table<CellType> parseTable()
    {
        // Parse the table start marker.
        parseTableStartMarker();

        // Parse the table header.
        TableHeader header = parseTableHeader();

        // Check the scalar type.
        if ( header.scalarTypeID != getScalarTypeID<CellType>() )
            throw ParseError( "Table has incompatible scalar type." );

        // Allocate a table and parse the data.
        Table<CellType> result( header.rowCount, header.columnCount );
        result.readCellData( m_stream );

        // Parse the table end marker.
        parseTableEndMarker();

        // Return the result.
        return result;
    }

    template <typename CellType>
    Table<CellType> parseTableAs()
    {
        // Parse the table start marker.
        parseTableStartMarker();

        // Parse the table header.
        TableHeader header = parseTableHeader();

        // Allocate a table.
        Table<CellType> result( header.rowCount, header.columnCount );

        // Read the table, convert if necessary.
        auto sourceType      = header.scalarTypeID;
        auto destinationType = getScalarTypeID<CellType>();
        if ( destinationType == sourceType )
        {
            // No conversion is necessary if source and destination types are the same.
            result.readCellData( m_stream );
        }
        else if ( sourceType == getScalarTypeID<float>() )
        {
            // Read as floats, convert to target type.
            result.template readCellDataAs<float>( m_stream );
        }
        else if ( sourceType == getScalarTypeID<int32_t>() )
        {
            // Read as floats, convert to target type.
            result.template readCellDataAs<int32_t>( m_stream );
        }
        else if ( sourceType == getScalarTypeID<uint8_t>() )
        {
            // Read as floats, convert to target type.
            result.template readCellDataAs<uint8_t>( m_stream );
        }
        else
        {
            throw ParseError( "Unsupported type conversion." );
        }

        // Parse the table end marker.
        parseTableEndMarker();

        // Return the result.
        return result;
    }

private:

    void parseFileSignature();

    void parseForestStartMarker();
    void parseForestEndMarker();
    void parseTreeStartMarker();
    void parseTreeEndMarker();
    void parseTableStartMarker();
    void parseTableEndMarker();

    bool atTableOfType( ScalarTypeID typeID );
    bool atTreeOfType( FeatureTypeID typeID );

    ForestHeader parseForestHeader();
    TreeHeader   parseTreeHeader();
    TableHeader  parseTableHeader();

    std::ifstream               m_stream;
    std::streampos              m_treeOffset;
    unsigned int                m_fileMajorVersion;
    unsigned int                m_fileMinorVersion;
    std::optional<std::string>  m_creatorName;
    std::optional<unsigned int> m_creatorMajorVersion;
    std::optional<unsigned int> m_creatorMinorVersion;
    std::optional<unsigned int> m_creatorPatchVersion;
};

template <typename CellType>
Table<CellType> readTable( const std::string & filename )
{
    BalsaFileParser parser( filename );
    return parser.parseTable<CellType>();
}

template <typename CellType>
Table<CellType> readTableAs( const std::string & filename )
{
    BalsaFileParser parser( filename );
    return parser.parseTableAs<CellType>();
}

class BalsaFileWriter
{
public:

    BalsaFileWriter( const std::string & filename );

    void setCreatorName( const std::string & value );
    void setCreatorMajorVersion( unsigned char value );
    void setCreatorMinorVersion( unsigned char value );
    void setCreatorPatchVersion( unsigned char value );

    template <typename FeatureType>
    void enterForest( unsigned char classCount, unsigned char featureCount )
    {
        enterForest( classCount, featureCount, getFeatureTypeID<FeatureType>() );
    }

    void leaveForest();

    template <typename FeatureIterator, typename OutputIterator>
    void writeTree( const DecisionTreeClassifier<FeatureIterator, OutputIterator> & tree )
    {
        typedef typename DecisionTreeClassifier<FeatureIterator, OutputIterator>::FeatureType FeatureType;

        writeFileHeaderOnce();
        writeTreeStartMarker();
        writeTreeHeader( tree.m_classCount, tree.m_featureCount, getFeatureTypeID<FeatureType>() );
        writeTable( tree.m_leftChildID );
        writeTable( tree.m_rightChildID );
        writeTable( tree.m_splitFeatureID );
        writeTable( tree.m_splitValue );
        writeTable( tree.m_label );
        writeTreeEndMarker();
    }

    template <typename CellType>
    void writeTable( const Table<CellType> & table )
    {
        writeFileHeaderOnce();
        writeTableStartMarker();
        writeTableHeader( table.getRowCount(), table.getColumnCount(), getScalarTypeID<CellType>() );
        table.writeCellData( m_stream );
        writeTableEndMarker();
    }

private:

    void writeFileHeaderOnce();
    void writeFileSignature();
    void writeEndiannessMarker();
    void writeTreeStartMarker();
    void writeTreeEndMarker();
    void writeTableStartMarker();
    void writeTableEndMarker();
    void enterForest( unsigned char classCount, unsigned char featureCount, FeatureTypeID featureType );
    void writeForestHeader( unsigned char classCount, unsigned char featureCount, FeatureTypeID featureType );
    void writeTreeHeader( unsigned char classCount, unsigned char featureCount, FeatureTypeID featureType );
    void writeTableHeader( unsigned int rowCount, unsigned int columnCount, ScalarTypeID scalarType );

    std::ofstream                m_stream;
    bool                         m_insideForest;
    bool                         m_fileHeaderWritten;
    std::optional<std::string>   m_creatorName;
    std::optional<unsigned char> m_creatorMajorVersion;
    std::optional<unsigned char> m_creatorMinorVersion;
    std::optional<unsigned char> m_creatorPatchVersion;
};

template <>
ScalarTypeID getScalarTypeID<uint8_t>();
template <>
ScalarTypeID getScalarTypeID<uint16_t>();
template <>
ScalarTypeID getScalarTypeID<uint32_t>();
template <>
ScalarTypeID getScalarTypeID<int8_t>();
template <>
ScalarTypeID getScalarTypeID<int16_t>();
template <>
ScalarTypeID getScalarTypeID<int32_t>();
template <>
ScalarTypeID getScalarTypeID<float>();
template <>
ScalarTypeID getScalarTypeID<double>();
template <>
ScalarTypeID getScalarTypeID<bool>();

template <>
FeatureTypeID getFeatureTypeID<float>();
template <>
FeatureTypeID getFeatureTypeID<double>();

} // namespace balsa

#endif // FILEIO_H
