#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "datatypes.h"
#include "exceptions.h"
#include "fileio.h"
#include "table.h"
#include "modelevaluation.h"

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
           << "   balsa_merge <outfile-name> <balsa-model-file>+" << std::endl;
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

        // Parse the output file name,
        if ( !(args >> token) ) throw ParseError( getUsage() );
        options.outputFile = token;

        // Parse the input filenames.
        while ( args >> token ) options.modelFiles.pusb_back( token );

        // Return results.
        return options;
    }

  std::string              outputFile;
  std::vector<std::string> modelFiles;
};

} // namespace

int main( int argc, char ** argv )
{
    try
    {
        // Parse the command-line options.
        auto options = Options::parseOptions( argc, argv );


    }
    catch ( Exception & e )
    {
        std::cerr << e.getMessage() << std::endl;
        return EXIT_FAILURE;
    }

    // Finish.
    return EXIT_SUCCESS;
}
