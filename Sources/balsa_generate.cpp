#include <iostream>
#include <string>
#include <sstream>
#include <cassert>
#include <random>

#include "datagenerator.h"
#include "datatypes.h"
#include "table.h"
#include "exceptions.h"

namespace
{
class Options
{
public:

    Options():
    seed( 0 ),
    pointCount( 1000 )
    {
    }

    static std::string getUsage()
    {
        std::stringstream ss;
        ss << "Usage:" << std::endl
           << std::endl
           << "   balsa_generate [options] <datagen_infile> <point_outfile> <label_outfile>" << std::endl
           << std::endl
           << " Options:" << std::endl
           << std::endl
           << "   -p <points> : Sets the number of points (default is 1000)."             << std::endl
           << "   -s <seed>   : Sets the random seed for data generation (default is 0)." << std::endl;

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

            // Parse the options.
            if ( token == "-s" )
            {
                if ( !( args >> options.seed ) ) throw ParseError( "Missing parameter to -s option." );

            }
            if ( token == "-p" )
            {
                if ( !( args >> options.pointCount ) ) throw ParseError( "Missing parameter to -p option." );
            }
            else
            {
                throw ParseError( std::string( "Unknown option: " ) + token );
            }
        }

        // Parse the filenames.
        if ( token.size() == 0 ) throw ParseError( getUsage() );
        options.datagenFile = token;
        if ( !( args >> options.pointFile) ) throw ParseError( getUsage() );
        if ( !( args >> options.labelFile) ) throw ParseError( getUsage() );

        // Return results.
        return options;
    }

    std::string  datagenFile;
    std::string  pointFile;
    std::string  labelFile;
    unsigned int seed;
    unsigned int featureCount;
    unsigned int pointCount;
};


} // namespace

int main( int argc, char ** argv )
{
    try
    {
        // Parse the command-line options.
        auto options = Options::parseOptions( argc, argv );

        // Construct a data generator from the configuration file.
        std::ifstream in;
        in.open( options.datagenFile );
        auto gen = parseDataGenerator<double>( in, options.seed );

        // Generate a data- and label set.
        Table<double> points( 1 );
        Table<Label>  labels( 1 );
        gen->generate( options.pointCount, points, labels );

        // Write the output files.
        std::ofstream out;
        out.open( options.pointFile, std::ios::binary );
        points.serialize( out );
        out.close();
        out.open( options.labelFile, std::ios::binary );
        labels.serialize( out );
        out.close();
    }
    catch ( Exception & e )
    {
        std::cerr << e.getMessage() << std::endl;
        return EXIT_FAILURE;
    }

    // Finish.
    return EXIT_SUCCESS;
}
