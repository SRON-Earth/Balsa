#include <iostream>
#include <string>
#include <sstream>
#include <cassert>

#include "exceptions.h"


namespace
{
class Options
{
public:

    Options():
    threadCount( 1 ),
    maxPreload( 1 )
    {
    }

    static std::string getUsage()
    {
        std::stringstream ss;
        ss << "Usage:" << std::endl
           << std::endl
           << "   balsa_dataproc [options] <output_file>" << std::endl
           << std::endl
           << " Options:" << std::endl
           << std::endl
           << "   -s <seed>: Sets the random seed for data generation (default is random)." << std::endl;
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

        // Parse all flags.
        Options options;
        while ( args >> token )
        {
            // Stop if the token is not a flag.
            assert( token.size() );
            if ( token[0] != '-' ) break;

            // Parse the '-s <seed>' option.
            if ( token == "-s" )
            {
                if ( !( args >> options.threadCount ) ) throw ParseError( "Missing parameter to -d option." );
            }
            else
            {
                throw ParseError( std::string( "Unknown option: " ) + token );
            }
        }

        // Parse the filenames.
        if ( token.size() == 0 ) throw ParseError( getUsage() );

        // Return  results.
        return options;
    }

    std::string modelFile;
    std::string dataFile;
    std::string outputFile;
    unsigned int threadCount;
    unsigned int maxPreload;
};

} // namespace

int main( int argc, char ** argv )
{
    try
    {
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
