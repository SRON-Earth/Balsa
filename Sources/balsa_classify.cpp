#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "datatypes.h"
#include "exceptions.h"
#include "randomforestclassifier.h"
#include "serdes.h"
#include "table.h"
#include "timing.h"

using namespace balsa;

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
           << "   balsa_classify [options] <model file> <datapoint file> <output file>" << std::endl
           << std::endl
           << " Options:" << std::endl
           << std::endl
           << "   -t <thread count> :  Sets the number of threads (default is 1)." << std::endl
           << "   -p <preload count>: Sets the number of trees to preload (default is 1)." << std::endl;
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

            // Parse the '-t <threadcount>' option.
            if ( token == "-t" )
            {
                if ( !( args >> options.threadCount ) ) throw ParseError( "Missing parameter to -t option." );
            }
            else if ( token == "-p" )
            {
                if ( !( args >> options.maxPreload ) ) throw ParseError( "Missing parameter to -p option." );
            }
            else
            {
                throw ParseError( std::string( "Unknown option: " ) + token );
            }
        }

        // Parse the filenames.
        if ( token.size() == 0 ) throw ParseError( getUsage() );
        options.modelFile = token;
        if ( !( args >> options.dataFile ) ) throw ParseError( getUsage() );
        if ( !( args >> options.outputFile ) ) throw ParseError( getUsage() );

        // Return  results.
        return options;
    }

    std::string  modelFile;
    std::string  dataFile;
    std::string  outputFile;
    unsigned int threadCount;
    unsigned int maxPreload;
};

} // namespace

int main( int argc, char ** argv )
{
    try
    {
        // Parse the command-line arguments.
        Options options = Options::parseOptions( argc, argv );

        // Debug.
        std::cout << "Model File : " << options.modelFile << std::endl;
        std::cout << "Data File  : " << options.dataFile << std::endl;
        std::cout << "Output File: " << options.outputFile << std::endl;
        std::cout << "Threads    : " << options.threadCount << std::endl;
        std::cout << "Preload    : " << options.maxPreload << std::endl;
        std::cout << std::endl;
        assert( options.threadCount > 0 );

        // Load the data.
        StopWatch watch;
        std::cout << "Ingesting data..." << std::endl;
        watch.start();
        auto dataSet = Table<double>::readFileAs( options.dataFile );
        std::cout << "Dataset loaded: " << dataSet.getColumnCount() << " features x " << dataSet.getRowCount() << " points." << std::endl;
        const auto dataLoadTime = watch.getElapsedTime();

        // Classify the data points.
        watch.start();
        Table<Label> labels( dataSet.getRowCount(), 1 );
        std::cout << labels.getRowCount() << " before " << std::endl;
        RandomForestClassifier<decltype( dataSet )::ConstIterator, decltype( labels )::Iterator> classifier( options.modelFile, options.threadCount - 1, options.maxPreload );
        classifier.classify( dataSet.begin(), dataSet.end(), dataSet.getColumnCount(), labels.begin() );
        std::cout << labels.getRowCount() << " after " << std::endl;
        watch.stop();
        const auto classificationTime = watch.getElapsedTime();

        // Store the labels.
        watch.start();
        std::ofstream outFile( options.outputFile, std::ios::binary );
        labels.serialize( outFile );
        watch.stop();
        const auto labelStoreTime = watch.getElapsedTime();

        std::cout << "Timings:" << std::endl
                  << "Data Load Time: " << dataLoadTime << std::endl
                  << "Classification Time: " << classificationTime << std::endl
                  << "Label Store Time: " << labelStoreTime << std::endl;
    }
    catch ( Exception & e )
    {
        std::cerr << e.getMessage() << std::endl;
        return EXIT_FAILURE;
    }

    // Finish.
    return EXIT_SUCCESS;
}
