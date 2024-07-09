#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "config.h"
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
           << "   balsa_convert <csv file> <output file>" << std::endl
           << std::endl
           << "Converts comma separated values (CSV) to double precision Balsa input files.";
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

        // Parse the filenames.
        if ( token.size() == 0 ) throw ParseError( getUsage() );
        options.csvFile = token;

        if ( !( args >> token ) ) throw ParseError( getUsage() );
        options.outputFile = token;

        // Return results.
        return options;
    }

    std::string csvFile;
    std::string outputFile;
};

} // namespace

int main( int argc, char ** argv )
{
    try
    {
        // Parse the command-line options.
        auto options = Options::parseOptions( argc, argv );

        // Parse the input file.
        std::ifstream in;
        in.open( options.csvFile );
        auto table = parseCSV<double>( in );

        // Write the output file.
        BalsaFileWriter fileWriter( options.outputFile, "balsa_convert",
            balsa_VERSION_MAJOR, balsa_VERSION_MINOR, balsa_VERSION_PATCH );
        fileWriter.writeTable( table );
    }
    catch ( Exception & e )
    {
        std::cerr << e.getMessage() << std::endl;
        return EXIT_FAILURE;
    }

    // Finish.
    return EXIT_SUCCESS;
}
