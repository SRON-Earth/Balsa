#include <map>
#include <variant>

#include "fileio.h"
#include "serdes.h"

namespace balsa
{

/*
 * Balsa file format version.
 */
constexpr const unsigned char FILE_FORMAT_MAJOR_VERSION = 1;
constexpr const unsigned char FILE_FORMAT_MINOR_VERSION = 0;

/*
 * Marker names.
 */
const std::string FILE_SIGNATURE          = "blsa";
const std::string BIG_ENDIAN_MARKER       = "bend";
const std::string LITTLE_ENDIAN_MARKER    = "lend";
const std::string FOREST_START_MARKER     = "frst";
const std::string FOREST_END_MARKER       = "tsrf";
const std::string TREE_START_MARKER       = "tree";
const std::string TREE_END_MARKER         = "eert";
const std::string TABLE_START_MARKER      = "tabl";
const std::string TABLE_END_MARKER        = "lbat";
const std::string DICTIONARY_START_MARKER = "dict";
const std::string DICTIONARY_END_MARKER   = "tcid";

/*
 * Dictionary key names.
 */
const std::string FILE_HEADER_FILE_MAJOR_VERSION_KEY    = "file_major_version";
const std::string FILE_HEADER_FILE_MINOR_VERSION_KEY    = "file_minor_version";
const std::string FILE_HEADER_CREATOR_NAME_KEY          = "creator_name";
const std::string FILE_HEADER_CREATOR_MINOR_VERSION_KEY = "creator_major_version";
const std::string FILE_HEADER_CREATOR_MAJOR_VERSION_KEY = "creator_minor_version";
const std::string FILE_HEADER_CREATOR_PATCH_VERSION_KEY = "creator_patch_version";
const std::string FOREST_HEADER_CLASS_COUNT_KEY         = "class_count";
const std::string FOREST_HEADER_FEATURE_COUNT_KEY       = "feature_count";
const std::string FOREST_HEADER_FEATURE_TYPE_ID_KEY     = "feature_type_id";
const std::string TREE_HEADER_CLASS_COUNT_KEY           = FOREST_HEADER_CLASS_COUNT_KEY;
const std::string TREE_HEADER_FEATURE_COUNT_KEY         = FOREST_HEADER_FEATURE_COUNT_KEY;
const std::string TREE_HEADER_FEATURE_TYPE_ID_KEY       = FOREST_HEADER_FEATURE_TYPE_ID_KEY;
const std::string TABLE_HEADER_ROW_COUNT_KEY            = "row_count";
const std::string TABLE_HEADER_COLUMN_COUNT_KEY         = "column_count";
const std::string TABLE_HEADER_SCALAR_TYPE_ID_KEY       = "scalar_type_id";

/*
 * An enumeration of recognized platform endianness.
 */
enum class Endianness
{
    BIG,
    LITTLE
};

/*
 * Returns the type name of the specified elementary type.
 */
template <typename Type>
std::string getTypeName()
{
    static_assert( sizeof( Type ) != sizeof( Type ), "Unsupported type." );
    return "";
}

// Template specializations for all supported elementary types.
template <> std::string getTypeName<uint8_t    >() { return "ui08"; }
template <> std::string getTypeName<uint16_t   >() { return "ui16"; }
template <> std::string getTypeName<uint32_t   >() { return "ui32"; }
template <> std::string getTypeName<int8_t     >() { return "in08"; }
template <> std::string getTypeName<int16_t    >() { return "in16"; }
template <> std::string getTypeName<int32_t    >() { return "in32"; }
template <> std::string getTypeName<float      >() { return "fl32"; }
template <> std::string getTypeName<double     >() { return "fl64"; }
template <> std::string getTypeName<bool       >() { return "bool"; }
template <> std::string getTypeName<std::string>() { return "strn"; }

/*
 * Returns the type name of the specified scalar type.
 */
std::string getTypeName( ScalarTypeID scalarTypeID )
{
    switch ( scalarTypeID )
    {
        case ScalarTypeID::UINT8:
            return getTypeName<uint8_t>();
        case ScalarTypeID::UINT16:
            return getTypeName<uint16_t>();
        case ScalarTypeID::UINT32:
            return getTypeName<uint32_t>();
        case ScalarTypeID::INT8:
            return getTypeName<int8_t>();
        case ScalarTypeID::INT16:
            return getTypeName<int16_t>();
        case ScalarTypeID::INT32:
            return getTypeName<int32_t>();
        case ScalarTypeID::FLOAT:
            return getTypeName<float>();
        case ScalarTypeID::DOUBLE:
            return getTypeName<double>();
        case ScalarTypeID::BOOL:
            return getTypeName<bool>();
        default:
            assert( false );
    }
}

/*
 * Returns the scalar type identifier that corresponds to the specified type
 * name.
 */
ScalarTypeID getScalarTypeID( const std::string & typeName )
{
    if ( typeName == getTypeName<uint8_t>() ) return ScalarTypeID::UINT8;
    if ( typeName == getTypeName<uint16_t>() ) return ScalarTypeID::UINT16;
    if ( typeName == getTypeName<uint32_t>() ) return ScalarTypeID::UINT32;
    if ( typeName == getTypeName<int8_t>() ) return ScalarTypeID::INT8;
    if ( typeName == getTypeName<int16_t>() ) return ScalarTypeID::INT16;
    if ( typeName == getTypeName<int32_t>() ) return ScalarTypeID::INT32;
    if ( typeName == getTypeName<float>() ) return ScalarTypeID::FLOAT;
    if ( typeName == getTypeName<double>() ) return ScalarTypeID::DOUBLE;
    if ( typeName == getTypeName<bool>() ) return ScalarTypeID::BOOL;
    throw ParseError( "Unknown scalar type: '" + typeName + "'." );
}

/*
 * Returns the type name of the specified feature type.
 */
std::string getTypeName( FeatureTypeID featureTypeID )
{
    switch ( featureTypeID )
    {
        case FeatureTypeID::FLOAT:
            return getTypeName<float>();
        case FeatureTypeID::DOUBLE:
            return getTypeName<double>();
        default:
            assert( false );
    }
}

/*
 * Returns the feature type identifier that corresponds to the specified type
 * name.
 */
FeatureTypeID getFeatureTypeID( const std::string & typeName )
{
    if ( typeName == getTypeName<float>() ) return FeatureTypeID::FLOAT;
    if ( typeName == getTypeName<double>() ) return FeatureTypeID::DOUBLE;
    throw ParseError( "Unknown feature type: '" + typeName + "'." );
}

/*
 * Serialize a string to a binary output stream.
 */
void serializeString( std::ostream & stream, const std::string & value )
{
    assert( value.size() < 256 );
    serialize( stream, static_cast<uint8_t>( value.size() ) );
    stream.write( value.data(), value.size() );
}

/*
 * Deserialize a string from a binary input stream.
 */
std::string deserializeString( std::istream & stream )
{
    uint8_t     length = deserialize<uint8_t>( stream );
    std::string value;
    value.resize( length );
    stream.read( value.data(), length );
    return value;
}

/*
 * A dictionary of which the values can be of any of the supported elementary
 * types.
 */
class Dictionary
{
    typedef std::string                                                                                           KeyType;
    typedef std::variant<uint8_t, uint16_t, uint32_t, int8_t, int16_t, int32_t, float, double, bool, std::string> ValueType;

public:

    /*
     * Returns the number of items in the dictionary.
     */
    std::size_t size() const
    {
        return m_dictionary.size();
    }

    /*
     * Enters the specified key into the dictionary with the specified value. If
     * the dictionary already contains the specified key the associated value
     * will be replaced by the specfied value.
     */
    template <typename T>
    void set( const std::string & key, const T & value )
    {
        auto & variant = m_dictionary[key];
        variant        = value;
        assert( std::holds_alternative<T>( variant ) );
        assert( m_dictionary.size() < 256 );
    }

    /*
     * Retrieves the value associated with the specified key from the
     * dictionary.
     */
    template <typename T>
    const T & get( const std::string & key ) const
    {
        return std::get<T>( m_dictionary.at( key ) );
    }

    /*
     * Returns the value associated with the specified key, or an empty value
     * if the dictionary does not contain the specified key.
     */
    template <typename T>
    std::optional<T> find( const std::string & key ) const
    {
        auto it = m_dictionary.find( key );
        if ( it == m_dictionary.end() ) return std::nullopt;
        return std::get<T>( it->second );
    }

    /*
     * Serialize the dictionary to a binary output stream.
     */
    void serialize( std::ostream & stream ) const
    {
        stream.write( DICTIONARY_START_MARKER.data(), DICTIONARY_START_MARKER.size() );
        balsa::serialize( stream, static_cast<uint8_t>( m_dictionary.size() ) );
        for ( const auto & keyValuePair : m_dictionary )
            std::visit( [&]( const auto & arg )
                {
                    serializeKVPair( stream, keyValuePair.first, arg );
                },
                keyValuePair.second );
        stream.write( DICTIONARY_END_MARKER.data(), DICTIONARY_END_MARKER.size() );
    }

    /*
     * Deserialize a dictionary from a binary output stream.
     */
    static Dictionary deserialize( std::istream & stream )
    {
        // Parse dictionary start marker.
        expect( stream, DICTIONARY_START_MARKER, "Missing dictionary start marker." );

        // Deserialize dictionary.
        Dictionary dictionary;
        uint8_t    size = balsa::deserialize<uint8_t>( stream );
        for ( unsigned int i = 0; i < size; ++i )
        {
            // Deserialize the key.
            KeyType key = deserializeString( stream );

            // Deserialize the value.
            ValueType   value;
            std::string typeName = getFixedSizeToken( stream, 4 );
            if ( typeName == getTypeName<uint8_t>() )
                value = balsa::deserialize<uint8_t>( stream );
            else if ( typeName == getTypeName<uint16_t>() )
                value = balsa::deserialize<uint16_t>( stream );
            else if ( typeName == getTypeName<uint32_t>() )
                value = balsa::deserialize<uint32_t>( stream );
            else if ( typeName == getTypeName<int8_t>() )
                value = balsa::deserialize<int8_t>( stream );
            else if ( typeName == getTypeName<int16_t>() )
                value = balsa::deserialize<int16_t>( stream );
            else if ( typeName == getTypeName<int32_t>() )
                value = balsa::deserialize<int32_t>( stream );
            else if ( typeName == getTypeName<float>() )
                value = balsa::deserialize<float>( stream );
            else if ( typeName == getTypeName<double>() )
                value = balsa::deserialize<double>( stream );
            else if ( typeName == getTypeName<bool>() )
                value = balsa::deserialize<bool>( stream );
            else if ( typeName == getTypeName<std::string>() )
                value = deserializeString( stream );
            else
                throw ParseError( "Invalid type name '" + typeName + "'." );

            // Store (key, value)-pair in dictionary.
            dictionary.m_dictionary[key] = value;
        }

        // Parse dictionary end marker.
        expect( stream, DICTIONARY_END_MARKER, "Missing dictionary end marker." );

        return dictionary;
    }

private:

    template <typename T>
    void serializeKVPair( std::ostream & stream, const std::string & key, const T & value ) const
    {
        serializeString( stream, key );
        std::string typeName = getTypeName<T>();
        stream.write( typeName.data(), typeName.size() );
        balsa::serialize( stream, value );
    }

    void serializeKVPair( std::ostream & stream, const std::string & key, const std::string & value ) const
    {
        serializeString( stream, key );
        std::string typeName = getTypeName<std::string>();
        stream.write( typeName.data(), typeName.size() );
        serializeString( stream, value );
    }

    std::map<KeyType, ValueType> m_dictionary;
};

/*
 * Determines the platform endianness.
 */
Endianness getPlatformEndianness()
{
    const uint32_t        value       = 0x00000001;
    const void *          voidAddress = static_cast<const void *>( &value );
    const unsigned char * charAddress = static_cast<const unsigned char *>( voidAddress );
    return ( *charAddress == 0x01 ) ? Endianness::LITTLE : Endianness::BIG;
}

/*
 * Parse the endianness marker from a binary input stream.
 */
Endianness parseFileEndianness( std::istream & stream )
{
    std::string marker = getFixedSizeToken( stream, 4 );
    if ( marker == LITTLE_ENDIAN_MARKER ) return Endianness::LITTLE;
    if ( marker == BIG_ENDIAN_MARKER ) return Endianness::BIG;
    throw ParseError( "Invalid endianness marker." );
}

BalsaFileParser::BalsaFileParser( const std::string & filename )
{
    // Configure the file input stream to throw an exception on error.
    m_stream.exceptions( std::ifstream::eofbit | std::ifstream::failbit | std::ifstream::badbit );

    // Open the model file.
    m_stream.open( filename, std::ios::binary );

    parseFileSignature();

    Endianness fileEndianness = parseFileEndianness( m_stream );
    if ( fileEndianness != getPlatformEndianness() )
        throw SupplierError( "Endianness mismatch." );

    Dictionary   header           = Dictionary::deserialize( m_stream );
    unsigned int fileMajorVersion = header.get<uint8_t>( FILE_HEADER_FILE_MAJOR_VERSION_KEY );
    unsigned int fileMinorVersion = header.get<uint8_t>( FILE_HEADER_FILE_MINOR_VERSION_KEY );

    if ( fileMajorVersion != FILE_FORMAT_MAJOR_VERSION )
    {
        throw SupplierError( "File format major version number mismatch." );
    }

    if ( fileMinorVersion < FILE_FORMAT_MINOR_VERSION )
    {
        throw SupplierError( "File format minor version number mismatch." );
    }

    m_fileMajorVersion = fileMajorVersion;
    m_fileMinorVersion = fileMinorVersion;

    m_creatorName         = header.find<std::string>( FILE_HEADER_CREATOR_NAME_KEY );
    m_creatorMajorVersion = header.find<uint8_t>( FILE_HEADER_CREATOR_MAJOR_VERSION_KEY );
    m_creatorMinorVersion = header.find<uint8_t>( FILE_HEADER_CREATOR_MINOR_VERSION_KEY );
    m_creatorPatchVersion = header.find<uint8_t>( FILE_HEADER_CREATOR_PATCH_VERSION_KEY );
}

unsigned int BalsaFileParser::getFileMajorVersion() const
{
    return m_fileMajorVersion;
}

unsigned int BalsaFileParser::getFileMinorVersion() const
{
    return m_fileMinorVersion;
}

std::optional<std::string> BalsaFileParser::getCreatorName() const
{
    return m_creatorName;
}

std::optional<unsigned int> BalsaFileParser::getCreatorMajorVersion() const
{
    return m_creatorMajorVersion;
}

std::optional<unsigned int> BalsaFileParser::getCreatorMinorVersion() const
{
    return m_creatorMinorVersion;
}

std::optional<unsigned int> BalsaFileParser::getCreatorPatchVersion() const
{
    return m_creatorPatchVersion;
}

bool BalsaFileParser::atTable()
{
    return ( peekFixedSizeToken( m_stream, TABLE_START_MARKER.size() ) == TABLE_START_MARKER );
}

bool BalsaFileParser::atTableOfType( ScalarTypeID typeID )
{
    bool result   = false;
    auto position = m_stream.tellg();
    if ( getFixedSizeToken( m_stream, TABLE_START_MARKER.size() ) == TABLE_START_MARKER )
    {
        TableHeader header = parseTableHeader();
        result             = ( header.scalarTypeID == typeID );
    }
    m_stream.seekg( position );
    return result;
}

bool BalsaFileParser::atTree()
{
    return ( peekFixedSizeToken( m_stream, TREE_START_MARKER.size() ) == TREE_START_MARKER );
}

bool BalsaFileParser::atTreeOfType( FeatureTypeID typeID )
{
    bool result   = false;
    auto position = m_stream.tellg();
    if ( getFixedSizeToken( m_stream, TREE_START_MARKER.size() ) == TREE_START_MARKER )
    {
        TreeHeader header = parseTreeHeader();
        result            = ( header.featureTypeID == typeID );
    }
    m_stream.seekg( position );
    return result;
}

bool BalsaFileParser::atForest()
{
    return ( peekFixedSizeToken( m_stream, FOREST_START_MARKER.size() ) == FOREST_START_MARKER );
}

bool BalsaFileParser::atEndOfForest()
{
    return ( peekFixedSizeToken( m_stream, FOREST_END_MARKER.size() ) == FOREST_END_MARKER );
}

bool BalsaFileParser::atEOF()
{
    return m_stream.peek() == EOF;
}

ForestHeader BalsaFileParser::enterForest()
{
    expect( m_stream, FOREST_START_MARKER, "Missing forest start marker." );
    ForestHeader result = parseForestHeader();
    m_treeOffset        = m_stream.tellg();
    return result;
}

void BalsaFileParser::leaveForest()
{
    expect( m_stream, FOREST_END_MARKER, "Missing forest end marker." );
}

void BalsaFileParser::reenterForest()
{
    if ( m_treeOffset == 0 ) throw ClientError( "No forrest was entered yet." );
    m_stream.seekg( m_treeOffset );
}

void BalsaFileParser::parseFileSignature()
{
    expect( m_stream, FILE_SIGNATURE, "Invalid file signature." );
}

void BalsaFileParser::parseTreeStartMarker()
{
    expect( m_stream, TREE_START_MARKER, "Missing tree start tag." );
}

void BalsaFileParser::parseTreeEndMarker()
{
    expect( m_stream, TREE_END_MARKER, "Missing tree end tag." );
}

void BalsaFileParser::parseTableStartMarker()
{
    expect( m_stream, TABLE_START_MARKER, "Invalid table start marker." );
}

void BalsaFileParser::parseTableEndMarker()
{
    expect( m_stream, TABLE_END_MARKER, "Invalid table end marker." );
}

ForestHeader BalsaFileParser::parseForestHeader()
{
    ForestHeader result;
    Dictionary   dictionary = Dictionary::deserialize( m_stream );
    result.classCount       = dictionary.get<uint8_t>( FOREST_HEADER_CLASS_COUNT_KEY );
    result.featureCount     = dictionary.get<uint8_t>( FOREST_HEADER_FEATURE_COUNT_KEY );
    result.featureTypeID    = getFeatureTypeID( dictionary.get<std::string>( FOREST_HEADER_FEATURE_TYPE_ID_KEY ) );
    return result;
}

TreeHeader BalsaFileParser::parseTreeHeader()
{
    TreeHeader result;
    Dictionary dictionary = Dictionary::deserialize( m_stream );
    result.classCount     = dictionary.get<uint8_t>( TREE_HEADER_CLASS_COUNT_KEY );
    result.featureCount   = dictionary.get<uint8_t>( TREE_HEADER_FEATURE_COUNT_KEY );
    result.featureTypeID  = getFeatureTypeID( dictionary.get<std::string>( TREE_HEADER_FEATURE_TYPE_ID_KEY ) );
    return result;
}

TableHeader BalsaFileParser::parseTableHeader()
{
    TableHeader result;
    Dictionary  dictionary = Dictionary::deserialize( m_stream );
    result.rowCount        = dictionary.get<uint32_t>( TABLE_HEADER_ROW_COUNT_KEY );
    result.columnCount     = dictionary.get<uint32_t>( TABLE_HEADER_COLUMN_COUNT_KEY );
    result.scalarTypeID    = getScalarTypeID( dictionary.get<std::string>( TABLE_HEADER_SCALAR_TYPE_ID_KEY ) );
    return result;
}

BalsaFileWriter::BalsaFileWriter( const std::string & filename ):
m_insideForest( false ),
m_fileHeaderWritten( false )
{
    // Configure the file input stream to throw an exception on error.
    m_stream.exceptions( std::ofstream::eofbit | std::ofstream::failbit | std::ofstream::badbit );

    // Open the file to write and truncate it if it exists.
    m_stream.open( filename, std::ios::binary );
}

void BalsaFileWriter::setCreatorName( const std::string & value )
{
    m_creatorName = value;
}

void BalsaFileWriter::setCreatorMajorVersion( unsigned char value )
{
    m_creatorMajorVersion = value;
}

void BalsaFileWriter::setCreatorMinorVersion( unsigned char value )
{
    m_creatorMinorVersion = value;
}

void BalsaFileWriter::setCreatorPatchVersion( unsigned char value )
{
    m_creatorPatchVersion = value;
}

void BalsaFileWriter::enterForest( unsigned char classCount, unsigned char featureCount, FeatureTypeID featureType )
{
    assert( !m_insideForest );
    writeFileHeaderOnce();
    m_stream.write( FOREST_START_MARKER.data(), FOREST_START_MARKER.size() );
    writeForestHeader( classCount, featureCount, featureType );
    m_insideForest = true;
}

void BalsaFileWriter::leaveForest()
{
    assert( m_insideForest );
    m_stream.write( FOREST_END_MARKER.data(), FOREST_END_MARKER.size() );
    m_insideForest = false;
}

void BalsaFileWriter::writeForestHeader( unsigned char classCount, unsigned char featureCount, FeatureTypeID featureType )
{
    Dictionary header;
    header.set<uint8_t>( FOREST_HEADER_CLASS_COUNT_KEY, classCount );
    header.set<uint8_t>( FOREST_HEADER_FEATURE_COUNT_KEY, featureCount );
    header.set<std::string>( FOREST_HEADER_FEATURE_TYPE_ID_KEY, getTypeName( featureType ) );
    header.serialize( m_stream );
}

void BalsaFileWriter::writeTreeHeader( unsigned char classCount, unsigned char featureCount, FeatureTypeID featureType )
{
    Dictionary header;
    header.set<uint8_t>( TREE_HEADER_CLASS_COUNT_KEY, classCount );
    header.set<uint8_t>( TREE_HEADER_FEATURE_COUNT_KEY, featureCount );
    header.set<std::string>( TREE_HEADER_FEATURE_TYPE_ID_KEY, getTypeName( featureType ) );
    header.serialize( m_stream );
}

void BalsaFileWriter::writeTableHeader( unsigned int rowCount, unsigned int columnCount, ScalarTypeID scalarType )
{
    Dictionary header;
    header.set<uint32_t>( TABLE_HEADER_ROW_COUNT_KEY, rowCount );
    header.set<uint32_t>( TABLE_HEADER_COLUMN_COUNT_KEY, columnCount );
    header.set<std::string>( TABLE_HEADER_SCALAR_TYPE_ID_KEY, getTypeName( scalarType ) );
    header.serialize( m_stream );
}

void BalsaFileWriter::writeFileHeaderOnce()
{
    if ( m_fileHeaderWritten ) return;

    writeFileSignature();
    writeEndiannessMarker();
    Dictionary dictionary;
    dictionary.set<uint8_t>( FILE_HEADER_FILE_MAJOR_VERSION_KEY, FILE_FORMAT_MAJOR_VERSION );
    dictionary.set<uint8_t>( FILE_HEADER_FILE_MINOR_VERSION_KEY, FILE_FORMAT_MINOR_VERSION );
    if ( m_creatorName ) dictionary.set<std::string>( FILE_HEADER_CREATOR_NAME_KEY, *m_creatorName );
    if ( m_creatorMajorVersion ) dictionary.set<uint8_t>( FILE_HEADER_CREATOR_MAJOR_VERSION_KEY, *m_creatorMajorVersion );
    if ( m_creatorMinorVersion ) dictionary.set<uint8_t>( FILE_HEADER_CREATOR_MINOR_VERSION_KEY, *m_creatorMinorVersion );
    if ( m_creatorPatchVersion ) dictionary.set<uint8_t>( FILE_HEADER_CREATOR_PATCH_VERSION_KEY, *m_creatorPatchVersion );
    dictionary.serialize( m_stream );

    m_fileHeaderWritten = true;
}

void BalsaFileWriter::writeFileSignature()
{
    m_stream.write( FILE_SIGNATURE.data(), FILE_SIGNATURE.size() );
}

void BalsaFileWriter::writeEndiannessMarker()
{
    Endianness          endianness = getPlatformEndianness();
    const std::string & marker     = ( endianness == Endianness::BIG ) ? BIG_ENDIAN_MARKER : LITTLE_ENDIAN_MARKER;
    m_stream.write( marker.data(), marker.size() );
}

void BalsaFileWriter::writeTreeStartMarker()
{
    m_stream.write( TREE_START_MARKER.data(), TREE_START_MARKER.size() );
}

void BalsaFileWriter::writeTreeEndMarker()
{
    m_stream.write( TREE_END_MARKER.data(), TREE_END_MARKER.size() );
}

void BalsaFileWriter::writeTableStartMarker()
{
    m_stream.write( TABLE_START_MARKER.data(), TABLE_START_MARKER.size() );
}

void BalsaFileWriter::writeTableEndMarker()
{
    m_stream.write( TABLE_END_MARKER.data(), TABLE_END_MARKER.size() );
}

template <>
ScalarTypeID getScalarTypeID<uint8_t>()
{
    return ScalarTypeID::UINT8;
}

template <>
ScalarTypeID getScalarTypeID<uint16_t>()
{
    return ScalarTypeID::UINT16;
}

template <>
ScalarTypeID getScalarTypeID<uint32_t>()
{
    return ScalarTypeID::UINT32;
}

template <>
ScalarTypeID getScalarTypeID<int8_t>()
{
    return ScalarTypeID::INT8;
}

template <>
ScalarTypeID getScalarTypeID<int16_t>()
{
    return ScalarTypeID::INT16;
}

template <>
ScalarTypeID getScalarTypeID<int32_t>()
{
    return ScalarTypeID::INT32;
}

template <>
ScalarTypeID getScalarTypeID<float>()
{
    return ScalarTypeID::FLOAT;
}

template <>
ScalarTypeID getScalarTypeID<double>()
{
    return ScalarTypeID::DOUBLE;
}

template <>
ScalarTypeID getScalarTypeID<bool>()
{
    return ScalarTypeID::BOOL;
}

template <>
FeatureTypeID getFeatureTypeID<float>()
{
    return FeatureTypeID::FLOAT;
}

template <>
FeatureTypeID getFeatureTypeID<double>()
{
    return FeatureTypeID::DOUBLE;
}

} // namespace balsa
