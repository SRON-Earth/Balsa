#include <cassert>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>

#include "datatypes.h"
#include "exceptions.h"
#include "fileio.h"
#include "table.h"

using namespace balsa;

namespace
{
class Options
{
public:

    Options()
    {
    }

    static std::string getUsage()
    {
        std::stringstream ss;
        ss << "Usage:" << std::endl
           << std::endl
           << "   balsa_print <balsa_file>" << std::endl;
        return ss.str();
    }

    static Options parseOptions( int argc, char ** argv )
    {
        // Put all arguments in a stringstream.
        std::stringstream args;
        for ( int i = 0; i < argc; ++i ) args << ' ' << argv[i];

        // Discard the executable name.
        std::string token;
        args >> token;
        token = "";

        // Parse all flags.
        Options options;
        while ( args >> token )
        {
            // Stop if the token is not a flag.
            assert( token.size() );
            if ( token[0] != '-' ) break;
            throw ParseError( std::string( "Unknown option: " ) + token );
        }

        // Parse the filename.
        if ( token.size() == 0 ) throw ParseError( getUsage() );
        options.fileName = token;

        // Return results.
        return options;
    }

    std::string fileName;
};

template <typename Type>
std::string getCommonTypeName()
{
    static_assert( sizeof( Type ) != sizeof( Type ), "Unsupported type." );
    return "";
}

template <> std::string getCommonTypeName<uint8_t>() { return "unsigned 8-bit integers"; }
template <> std::string getCommonTypeName<uint16_t>() { return "unsigned 16-bit integers"; }
template <> std::string getCommonTypeName<uint32_t>() { return "unsigned 32-bit integers"; }
template <> std::string getCommonTypeName<int8_t>() { return "signed 8-bit integers"; }
template <> std::string getCommonTypeName<int16_t>() { return "signed 16-bit integers"; }
template <> std::string getCommonTypeName<int32_t>() { return "signed 32-bit integers"; }
template <> std::string getCommonTypeName<bool>() { return "booleans"; }
template <> std::string getCommonTypeName<float>() { return "single precision floating point numbers"; }
template <> std::string getCommonTypeName<double>() { return "double precision floating point numbers"; }

template <typename Type>
void parseAndPrintTable( BalsaFileParser & parser )
{
    // Parse the table.
    auto table = parser.parseTable<Type>();

    // Print the header.
    std::cout << "TABLE " << table.getRowCount() << " rows x " << table.getColumnCount() << " columns of " << getCommonTypeName<Type>() << std::endl;

    // Print the values.
    std::cout << table;
}

template <typename Type>
void parseAndPrintTree( BalsaFileParser & parser )
{
    // Parse the tree.
    auto data = parser.parseTreeData<Type>();

    // Print the header.
    std::cout << "TREE " << data.classCount << " classes, " << data.featureCount << " features." << std::endl;

    // Print the values.
    std::cout << "N:   L:   R:   F:   V:              L:" << std::endl;
    for ( unsigned int row = 0; row < data.leftChildID.getRowCount(); ++row )
    {
        std::cout << std::left << std::setw( 4 ) << row << " "
                  << std::left << std::setw( 4 ) << data.leftChildID( row, 0 ) << " " << std::setw( 4 ) << data.rightChildID( row, 0 ) << " "
                  << std::left << std::setw( 4 ) << static_cast<int>( data.splitFeatureID( row, 0 ) ) << " " << std::setw( 4 ) << std::setw( 16 ) << data.splitValue( row, 0 )
                  << std::left << std::setw( 4 ) << int( data.label( row, 0 ) ) << std::endl;
    }
}

} // namespace

int main( int argc, char ** argv )
{
    try
    {
        // Parse the command-line options.
        auto options = Options::parseOptions( argc, argv );

        // Open the input file.
        BalsaFileParser parser( options.fileName );

        // Print the file format version.
        std::cout << "File version   : " << parser.getFileMajorVersion() << "." << parser.getFileMinorVersion() << std::endl;

        // Print information about the tool that created the file.
        auto creatorName = parser.getCreatorName();
        auto creatorMajorVersion = parser.getCreatorMajorVersion();
        auto creatorMinorVersion = parser.getCreatorMinorVersion();
        auto creatorPatchVersion = parser.getCreatorPatchVersion();
        std::cout << "Creator name   : "
            << (creatorName ? *creatorName : "*** UNKNOWN ***") << std::endl;
        std::cout << "Creator version: "
            << (creatorMajorVersion ? std::to_string(*creatorMajorVersion) : "?") << "."
            << (creatorMinorVersion ? std::to_string(*creatorMinorVersion) : "?") << "."
            << (creatorPatchVersion ? std::to_string(*creatorPatchVersion) : "?") << std::endl;

        // Read and print data objects until the end of the file.
        while ( !parser.atEOF() )
        {
            // Print a newline.
            std::cout << std::endl;

            // Print the object at the current position in the file.
            if ( parser.atForest() )
            {
                // A forest is just a list of trees. Consume the marker and continue.
                ForestHeader header = parser.enterForest();
                std::cout << "FOREST " << header.classCount << " classes." << std::endl;
            }
            else if ( parser.atEndOfForest() )
            {
                std::cout << "END OF FOREST" << std::endl;
                parser.leaveForest();
            }
            else if ( parser.atTree() )
            {
                // Parse and print the tree.
                if      ( parser.atTreeOfType<float >() ) parseAndPrintTree<float >( parser );
                else if ( parser.atTreeOfType<double>() ) parseAndPrintTree<double>( parser );
                else assert( false );
            }
            else if ( parser.atTable() )
            {
                // Parse and print the table.
                if      ( parser.atTableOfType<uint8_t >() ) parseAndPrintTable<uint8_t >( parser );
                else if ( parser.atTableOfType<uint16_t>() ) parseAndPrintTable<uint16_t>( parser );
                else if ( parser.atTableOfType<uint32_t>() ) parseAndPrintTable<uint32_t>( parser );
                else if ( parser.atTableOfType<int8_t  >() ) parseAndPrintTable<int8_t  >( parser );
                else if ( parser.atTableOfType<int16_t >() ) parseAndPrintTable<int16_t >( parser );
                else if ( parser.atTableOfType<int32_t >() ) parseAndPrintTable<int32_t >( parser );
                else if ( parser.atTableOfType<float   >() ) parseAndPrintTable<float   >( parser );
                else if ( parser.atTableOfType<double  >() ) parseAndPrintTable<double  >( parser );
                else if ( parser.atTableOfType<bool    >() ) parseAndPrintTable<bool    >( parser );
                else assert( false );
            }
        }
    }
    catch ( Exception & e )
    {
        std::cerr << e.getMessage() << std::endl;
        return EXIT_FAILURE;
    }

    // Finish.
    return EXIT_SUCCESS;
}
