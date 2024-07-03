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

/*
 * An enumeration of supported scalar types.
 */
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

/*
 * Returns the scalar type identifier for the specified type.
 */
template <typename Type>
ScalarTypeID getScalarTypeID()
{
    static_assert( sizeof( Type ) != sizeof( Type ), "Unsupported scalar type." );
    return static_cast<ScalarTypeID>( 0 );
}

/*
 * An enumeration of supported feature types.
 */
enum class FeatureTypeID
{
    FLOAT,
    DOUBLE
};

/*
 * Returns the feature type identifier for the specified type.
 */
template <typename Type>
FeatureTypeID getFeatureTypeID()
{
    static_assert( sizeof( Type ) != sizeof( Type ), "Unsupported feature type." );
    return static_cast<FeatureTypeID>( 0 );
}

/*
 * Description of a forest (an ensemble of decision trees).
 */
struct ForestHeader
{
    unsigned char classCount;     // Number of classes distinguished by the forest.
    unsigned char featureCount;   // Number of features the forest was trained on.
    FeatureTypeID featureTypeID;  // Numeric type used for features.
};

/*
 * Description of a decision tree.
 */
struct TreeHeader
{
    unsigned char classCount;     // Number of classes distinguished by the tree.
    unsigned char featureCount;   // Number of features the tree was trained on.
    FeatureTypeID featureTypeID;  // Numeric type used for features.
};

/*
 * Description of a table.
 */
struct TableHeader
{
    unsigned int rowCount;      // Number of rows.
    unsigned int columnCount;   // Number of columns.
    ScalarTypeID scalarTypeID;  // Numeric type of the elements of the table.
};

/*
 * Internal representation of a decision tree.
 */
template <typename FeatureType>
struct TreeData
{
    unsigned int       classCount;
    unsigned int       featureCount;
    Table<NodeID>      leftChildID;
    Table<NodeID>      rightChildID;
    Table<FeatureID>   splitFeatureID;
    Table<FeatureType> splitValue;
    Table<Label>       label;
};

/*
 * A parser for files written in the balsa file format.
 */
class BalsaFileParser
{
public:

	/*
	 * Constructor; opens the specified file for parsing.
	 */
    BalsaFileParser( const std::string & filename );

    /*
     * Returns the major version number of the balsa file format specification
     * the file adheres to.
     */
    unsigned int getFileMajorVersion() const;

    /*
     * Returns the minor version number of the balsa file format specification
     * the file adheres to.
     */
    unsigned int getFileMinorVersion() const;

    /*
     * Returns the name of the tool that created the file (if available).
     */
    std::optional<std::string>  getCreatorName() const;

    /*
     * Returns the major version number of the tool that created the file
     * (if available).
     */
    std::optional<unsigned int> getCreatorMajorVersion() const;

    /*
     * Returns the minor version number of the tool that created the file
     * (if available).
     */
    std::optional<unsigned int> getCreatorMinorVersion() const;

    /*
     * Returns the patch version number of the tool that created the file
     * (if available).
     */
    std::optional<unsigned int> getCreatorPatchVersion() const;

    /*
     * Returns true iff the reader is positioned at the end of the file.
     */
    bool atEOF();

    /*
     * Returns true iff the reader is positioned at the start of a forest.
     */
    bool atForest();

    /*
     * Returns true iff the reader is positioned at end of a forest.
     */
    bool atEndOfForest();

    /*
     * Returns true iff the reader is positioned at a decision tree.
     */
    bool atTree();

    /*
     * Returns true iff the reader is positioned at a table.
     */
    bool atTable();

    /*
     * Returns true iff the reader is positioned at a decision tree using
     * features of the specified type.
     */
    template <typename FeatureType>
    bool atTreeOfType()
    {
        return atTreeOfType( getFeatureTypeID<FeatureType>() );
    }

    /*
     * Returns true iff the reader is positioned at a table that contains
     * elements of the specified type.
     */
    template <typename ScalarType>
    bool atTableOfType()
    {
        return atTableOfType( getScalarTypeID<ScalarType>() );
    }

    /*
     * Parses a forest start marker and description.
     *
     * \pre The parser is positioned at a forest.
     * \post The parser will be positioned at the first decision tree in the
     *  forest.
     * \post The \c reenterForest() member function can be used to reposition
     *  the parser at the first decision tree in the forest.
     * \returns Forest description.
     */
    ForestHeader enterForest();

    /*
     * Parses and discards a forest end marker.
     *
     * \pre The parser is positioned at the end of a forest.
     * \post The parser will be positioned at the next object in the file, or at
     *  the end of the file if it contains no more objects.
     */
    void leaveForest();

    /*
     * Reposition the parser at the first decision tree of the last forest
     * entered using \c enterForest().
     *
     * \pre A forest was entered using \c enterForest().
     */
    void reenterForest();

    /*
     * Parses a decision tree and returns its internal representation.
     *
     * \pre The parser is positioned at a decision tree of the specified feature
     *  type.
     * \post The parser will be positioned at the next object in the file, or at
     *  the end of the file if it contains no more objects.
     */
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

    /*
     * Parses a decision tree.
     *
     * \pre The parser is positioned at a decision tree. The feature type of the
     *  decision tree should match the value type of the specified \c
     *  FeatureIterator type.
     * \post The parser will be positioned at the next object in the file, or at
     *  the end of the file if it contains no more objects.
     */
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

    /*
     * Parses a table containing elements of the specified scalar type.
     *
     * \pre The parser is positioned at a table of the specified scalar type.
     * \post The parser will be positioned at the next object in the file, or at
     *  the end of the file if it contains no more objects.
     */
    template <typename ScalarType>
    Table<ScalarType> parseTable()
    {
        // Parse the table start marker.
        parseTableStartMarker();

        // Parse the table header.
        TableHeader header = parseTableHeader();

        // Check the scalar type.
        if ( header.scalarTypeID != getScalarTypeID<ScalarType>() )
            throw ParseError( "Table has incompatible scalar type." );

        // Allocate a table and parse the data.
        Table<ScalarType> result( header.rowCount, header.columnCount );
        result.readCellData( m_stream );

        // Parse the table end marker.
        parseTableEndMarker();

        // Return the result.
        return result;
    }

    /*
     * Parses a table containing elements of the specified scalar type. If the
     * table stored in the file contains elements of a different scalar type,
     * the elements will be converted to the requested type if possible.
     *
     * \pre The parser is positioned at a table that contains elements of a
     *  scalar type that can be converted to the requested type.
     * \post The parser will be positioned at the next object in the file, or at
     *  the end of the file if it contains no more objects.
     */
    template <typename ScalarType>
    Table<ScalarType> parseTableAs()
    {
        // Parse the table start marker.
        parseTableStartMarker();

        // Parse the table header.
        TableHeader header = parseTableHeader();

        // Allocate a table.
        Table<ScalarType> result( header.rowCount, header.columnCount );

        // Read the table, convert if necessary.
        auto sourceType      = header.scalarTypeID;
        auto destinationType = getScalarTypeID<ScalarType>();
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

/*
 * Read a table containing elements of the specified scalar type from a file.
 */
template <typename ScalarType>
Table<ScalarType> readTable( const std::string & filename )
{
    BalsaFileParser parser( filename );
    return parser.parseTable<ScalarType>();
}

/*
 * Read a table containing elements of the specified scalar type from a file. If
 * the table stored in the file contains elements of a different scalar type,
 * the elements will be converted to the requested type if possible.
 */
template <typename ScalarType>
Table<ScalarType> readTableAs( const std::string & filename )
{
    BalsaFileParser parser( filename );
    return parser.parseTableAs<ScalarType>();
}

/*
 * A writer for files that adhere to the balsa file format.
 */
class BalsaFileWriter
{
public:

	/*
	 * Constructor; opens the specified file for writing. The file will be
	 * truncated if it exists.
	 */
    BalsaFileWriter( const std::string & filename );

    /*
     * Set the name of the tool that created this file.
     *
     * This information will be stored in the file header. The creator name is
     * optional; if this function has not been called before writing the first
     * object to the file, no creator name will be written to the file header.
     */
    void setCreatorName( const std::string & value );

    /*
     * Set the major version number of the tool that created this file.
     *
     * This information will be stored in the file header. The creator major
     * version number is optional; if this function has not been called before
     * writing the first object to the file, no creator major version number
     * will be written to the file header.
     */
    void setCreatorMajorVersion( unsigned char value );

    /*
     * Set the minor version number of the tool that created this file.
     *
     * This information will be stored in the file header. The creator major
     * version number is optional; if this function has not been called before
     * writing the first object to the file, no creator minor version number
     * will be written to the file header.
     */
    void setCreatorMinorVersion( unsigned char value );

    /*
     * Set the patch version number of the tool that created this file.
     *
     * This information will be stored in the file header. The creator patch
     * version number is optional; if this function has not been called before
     * writing the first object to the file, no creator patch version number
     * will be written to the file header.
     */
    void setCreatorPatchVersion( unsigned char value );

    /*
     * Write a forest start marker and forest description.
     *
     * After calling this function, the decision trees that compose the forest
     * can be written using the \c writeTree() function. Once all decision
     * trees have been written, the forest should be finalized using a call to
     * the \c leaveForest() function.
     *
     * \pre The writer is not positioned inside a forest (forests cannot be
     *  nested).
     */
    template <typename FeatureType>
    void enterForest( unsigned char classCount, unsigned char featureCount )
    {
        enterForest( classCount, featureCount, getFeatureTypeID<FeatureType>() );
    }

    /*
     * Write a forest end marker.
     *
     * This function should be called after all decision trees that compose the
     * forest have been written.
     *
     * \pre The writer is positioned inside a forest.
     */
    void leaveForest();

    /*
     * Write a decision tree to the file.
     *
     * Decision trees can be written as part of a forest, or as top-level
     * objects.
     */
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

    /*
     * Write a table to the file.
     *
     * \pre The writer is not positioned inside a forest.
     */
    template <typename ScalarType>
    void writeTable( const Table<ScalarType> & table )
    {
        writeFileHeaderOnce();
        writeTableStartMarker();
        writeTableHeader( table.getRowCount(), table.getColumnCount(), getScalarTypeID<ScalarType>() );
        table.writeCellData( m_stream );
        writeTableEndMarker();
    }

private:

    void enterForest( unsigned char classCount, unsigned char featureCount, FeatureTypeID featureType );

    void writeForestHeader( unsigned char classCount, unsigned char featureCount, FeatureTypeID featureType );
    void writeTreeHeader( unsigned char classCount, unsigned char featureCount, FeatureTypeID featureType );
    void writeTableHeader( unsigned int rowCount, unsigned int columnCount, ScalarTypeID scalarType );

    void writeFileHeaderOnce();
    void writeFileSignature();
    void writeEndiannessMarker();
    void writeTreeStartMarker();
    void writeTreeEndMarker();
    void writeTableStartMarker();
    void writeTableEndMarker();

    std::ofstream                m_stream;
    bool                         m_insideForest;
    bool                         m_fileHeaderWritten;
    std::optional<std::string>   m_creatorName;
    std::optional<unsigned char> m_creatorMajorVersion;
    std::optional<unsigned char> m_creatorMinorVersion;
    std::optional<unsigned char> m_creatorPatchVersion;
};

// Template specialization for all supported scalar types.
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

// Template specializations for all supported feature types.
template <>
FeatureTypeID getFeatureTypeID<float>();
template <>
FeatureTypeID getFeatureTypeID<double>();

} // namespace balsa

#endif // FILEIO_H
