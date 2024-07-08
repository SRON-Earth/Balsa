#include <cassert>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>

#include "classifier.h"
#include "datatypes.h"
#include "decisiontreeclassifier.h"
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

class PrintDispatcher: public ClassifierVisitor
{
public:

    void visit( const EnsembleClassifier &classifier )
    {
        (void) classifier;
        assert( false );
    }

    void visit( const DecisionTreeClassifier<float> &classifier )
    {
        std::cout << classifier;
    }

    void visit( const DecisionTreeClassifier<double> &classifier )
    {
        std::cout << classifier;
    }
};

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
            if ( parser.atEnsemble() )
            {
                // An ensemble is just a list of classifiers. Consume the marker and continue.
                EnsembleHeader header = parser.enterEnsemble();
                std::cout << "ENSEMBLE " << static_cast<unsigned int>( header.classCount ) << " classes, "
                          << static_cast<unsigned int>( header.featureCount ) << " features." << std::endl;
            }
            else if ( parser.atEndOfEnsemble() )
            {
                std::cout << "END OF ENSEMBLE" << std::endl;
                parser.leaveEnsemble();
            }
            else if ( parser.atTree() )
            {
                PrintDispatcher printer;
                auto classifier = parser.parseClassifier();
                classifier->visit( printer );
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
