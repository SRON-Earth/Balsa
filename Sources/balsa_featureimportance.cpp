#include <cassert>
#include <iostream>
#include <random>
#include <sstream>
#include <string>

#include "randomforestclassifier.h"
#include "exceptions.h"
#include "serdes.h"
#include "table.h"
#include "modelevaluation.h"

using namespace balsa;

namespace
{
class Options
{
public:

    Options():
    repeatCount( 5 )
    {
    }

    static std::string getUsage()
    {
        std::stringstream ss;
        ss << "Usage:" << std::endl
           << std::endl
           << "   balsa_featureimportance [options] <model file> <data input file> <label input file>" << std::endl
           << std::endl
           << " Options:" << std::endl
           << std::endl
           << "   -r <repeats>     : Number of repeats used to determine feature importance (default: 5)." << std::endl;
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

            if ( token == "-r" )
            {
                if ( !( args >> options.repeatCount ) ) throw ParseError( "Missing parameter to -r option." );
                if ( options.repeatCount < 1 ) throw ParseError( "Repeat count must be positive." );
            }
        }

        // Parse the filenames.
        if ( token.size() == 0 ) throw ParseError( getUsage() );
        options.modelFile = token;
        if ( !( args >> options.dataFile  ) ) throw ParseError( "Missing data file."  );
        if ( !( args >> options.labelFile ) ) throw ParseError( "Missing label file." );

        // Return  results.
        return options;
    }

    std::string  modelFile;
    std::string  dataFile;
    std::string  labelFile;
    unsigned int repeatCount;
};
} // namespace

int main( int argc, char ** argv )
{
    try
    {
        // Parse the command-line arguments.
        Options options = Options::parseOptions( argc, argv );

        // Load the test data.
        auto dataSet = Table<double>::readFileAs( options.dataFile );
        auto labels  = Table<Label>::readFileAs( options.labelFile );

        // Create a classifier for the model.
        RandomForestClassifier< decltype( dataSet )::ConstIterator, decltype( labels )::Iterator > classifier( options.modelFile  );

        // Calculate the feature importance and print them.
        std::cout << "Analyzing feature importance..." << std::endl;
        FeatureImportances importances( classifier, dataSet.begin(), dataSet.end(), labels.begin(), dataSet.getColumnCount(), options.repeatCount );
        std::cout << "Done." << std::endl;
        std::cout << importances << std::endl;
    }
    catch ( Exception & e )
    {
        std::cerr << e.getMessage() << std::endl;
        return EXIT_FAILURE;
    }

    // Finish.
    return EXIT_SUCCESS;
}
