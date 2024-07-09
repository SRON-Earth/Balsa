#ifndef FILEIO_H
#define FILEIO_H

#include <fstream>
#include <optional>
#include <string>

#include "classifier.h"
#include "classifiervisitor.h"
#include "datatypes.h"
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
 * Description of an ensemble of classification models.
 */
struct EnsembleHeader
{
    unsigned char classCount;     // Number of classes distinguished by the ensemble.
    unsigned char featureCount;   // Number of features the ensemble was trained on.
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
     * Returns true iff the reader is positioned at the start of an ensemble.
     */
    bool atEnsemble();

    /*
     * Returns true iff the reader is positioned at end of an ensemble.
     */
    bool atEndOfEnsemble();

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
     * Parses an ensemble start marker and description.
     *
     * \pre The parser is positioned at an ensemble.
     * \post The parser will be positioned at the first submodel in the
     *  ensemble.
     * \post The \c reenterEnsemble() member function can be used to reposition
     *  the parser at the first submodel in the ensemble.
     * \returns Ensemble description.
     */
    EnsembleHeader enterEnsemble();

    /*
     * Parses and discards an ensemble end marker.
     *
     * \pre The parser is positioned at the end of an ensemble.
     * \post The parser will be positioned at the next object in the file, or at
     *  the end of the file if it contains no more objects.
     */
    void leaveEnsemble();

    /*
     * Reposition the parser at the first submodel of the last ensemble
     * entered using \c enterEnsemble().
     *
     * \pre An ensemble was entered using \c enterEnsemble().
     */
    void reenterEnsemble();

    /*
     * Parses a classifier.
     *
     * \pre The parser is positioned at a classifier.
     * \post The parser will be positioned at the next object in the file, or at
     *  the end of the file if it contains no more objects.
     */
    Classifier::SharedPointer parseClassifier();

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
    void parseEnsembleStartMarker();
    void parseEnsembleEndMarker();
    void parseTreeStartMarker();
    void parseTreeEndMarker();
    void parseTableStartMarker();
    void parseTableEndMarker();

    bool atTableOfType( ScalarTypeID typeID );
    bool atTreeOfType( FeatureTypeID typeID );

    EnsembleHeader parseEnsembleHeader();
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
     *
     * \param filename Name of the file to write.
     * \param creatorName Name of the tool that created the file (optional).
     *  This information will be stored in the file header.
     * \param creatorMajorVersion Major version number of the tool that created
     *  the file (optional). This information will be stored in the file header.
     * \param creatorMinorVersion Minor version number of the tool that created
     *  the file (optional). This information will be stored in the file header.
     * \param creatorPatchVersion Patch version number of the tool that created
     *  the file (optional). This information will be stored in the file header.
	 */
    BalsaFileWriter( const std::string & filename,
        std::optional<std::string> creatorName = std::nullopt,
        std::optional<unsigned char> creatorMajorVersion = std::nullopt,
        std::optional<unsigned char> creatorMinorVersion = std::nullopt,
        std::optional<unsigned char> creatorPatchVersion = std::nullopt );

    /*
     * Write an ensemble start marker and ensemble description.
     *
     * After calling this function, the submodels that compose the ensemble can
     * be written using the \c writeTree() function. Once all submodels have
     * been written, the ensemble should be finalized using a call to the \c
     * leaveEnsemble() function.
     *
     * \pre The writer is not positioned inside an ensemble (ensembles cannot be
     *  nested).
     */
    void enterEnsemble( unsigned char classCount, unsigned char featureCount );

    /*
     * Write an ensemble end marker.
     *
     * This function should be called after all submodels that compose the
     * ensemble have been written.
     *
     * \pre The writer is positioned inside an ensemble.
     */
    void leaveEnsemble();

    /*
     * Write a model to the file.
     *
     * Decision trees can be written as part of an ensemble, or as top-level
     * objects.
     */
    void writeClassifier( const Classifier & classifier );

    /*
     * Write a table to the file.
     *
     * \pre The writer is not positioned inside an ensemble.
     */
    template <typename ScalarType>
    void writeTable( const Table<ScalarType> & table )
    {
        writeTableStartMarker();
        writeTableHeader( table.getRowCount(), table.getColumnCount(), getScalarTypeID<ScalarType>() );
        table.writeCellData( m_stream );
        writeTableEndMarker();
    }

private:

    class ClassifierWriteDispatcher: public ClassifierVisitor
    {
    public:

      ClassifierWriteDispatcher( BalsaFileWriter & writer ):
      m_writer( writer )
      {
      }

      void visit( const EnsembleClassifier &classifier );
      void visit( const DecisionTreeClassifier<float> &classifier );
      void visit( const DecisionTreeClassifier<double> &classifier );

    private:

        BalsaFileWriter & m_writer;

    };

    void writeFileSignature();
    void writeEndiannessMarker();
    void writeTreeStartMarker();
    void writeTreeEndMarker();
    void writeTableStartMarker();
    void writeTableEndMarker();
    void writeEnsembleHeader( unsigned char classCount, unsigned char featureCount );
    void writeTreeHeader( unsigned char classCount, unsigned char featureCount, FeatureTypeID featureType );
    void writeTableHeader( unsigned int rowCount, unsigned int columnCount, ScalarTypeID scalarType );

    std::ofstream m_stream;
    bool          m_insideEnsemble;
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
