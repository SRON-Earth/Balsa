#include <iostream>
#include <string>
#include <sstream>
#include <cassert>
#include <random>

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
    featureCount( 3 ),
    pointCount( 1000 ),
    labelCount( 2 )
    {
    }

    static std::string getUsage()
    {
        std::stringstream ss;
        ss << "Usage:" << std::endl
           << std::endl
           << "   balsa_generate [options] <point_outfile> <label_outfile>" << std::endl
           << std::endl
           << " Options:" << std::endl
           << std::endl
           << "   -s <seed>  : Sets the random seed for data generation (default is 0)." << std::endl
           << "   -f <count> : Sets the feature count (default is 3)."                   << std::endl
           << "   -p <points>: Sets the point count (default is 1000)."                  << std::endl
           << "   -l <labels>: Sets the label count (default is 2)."                     << std::endl;
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
                if ( !( args >> options.seed ) ) throw ParseError( "Missing parameter to -s option." );
            }
            else if ( token == "-f" )
            {
                if ( !( args >> options.featureCount ) ) throw ParseError( "Missing parameter to -f option." );
            }
            else if ( token == "-p" )
            {
                if ( !( args >> options.pointCount ) ) throw ParseError( "Missing parameter to -p option." );
            }
            else
            {
                throw ParseError( std::string( "Unknown option: " ) + token );
            }
        }

        // Parse the filename.
        if ( token.size() == 0 ) throw ParseError( getUsage() );
        options.pointFile = token;
        if ( !( args >> options.labelFile) ) throw ParseError( getUsage() );

        // Return results.
        return options;
    }

    std::string  pointFile;
    std::string  labelFile;
    unsigned int seed;
    unsigned int featureCount;
    unsigned int pointCount;
    unsigned int labelCount;
};

} // namespace

int main( int argc, char ** argv )
{
    try
    {
        // Parse the command-line options.
        auto options = Options::parseOptions( argc, argv );

        // Generate a table full of points.
        std::mt19937 gen( options.seed ); // mersenne_twister_engine seeded with rd()
        std::uniform_int_distribution<> distribution(0, 1000.0);
        Table<double> points( options.pointCount, options.featureCount );
        for ( auto it( points.begin() ), end( points.end() ); it != end; ++it )
        {
            *it = distribution(gen);
        }

        // Generate labels.
        Table<Label> labels( options.pointCount, 1 );
        std::uniform_int_distribution<> labelDistribution( 0, options.labelCount - 1 );
        for ( auto it ( labels.begin() ), end( labels.end()); it != end; ++it )
        {
            *it = labelDistribution(gen);
        }

        // Write the output files.
        std::ofstream out;
        out.open( options.pointFile, std::ios::binary );
        out << points;
        out.close();

        out.open( options.labelFile, std::ios::binary );
        out << labels;
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
