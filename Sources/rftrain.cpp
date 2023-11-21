#include <iostream>
#include <sstream>

#include "timing.h"
#include "exceptions.h"
#include "datamodel.h"

#include <string>
#include <vector>
#include <random>
#include <chrono>

#include "ingestion.h"

namespace
{
  class Options
  {
  public:

    Options():
    maxDepth   ( std::numeric_limits<unsigned int>::max() ),
    treeCount  ( 150                                      ),
    threadCount( 1                                        )
    {
    }

    static constexpr const char * getUsage()
    {
        return  "Usage: rftrain [-t <max threads>] <training input file> <model output file>";
    }

    static Options parseOptions( int argc, char **argv )
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
                if ( !(args >> options.threadCount) ) throw ParseError( "Missing parameter to -t option." );
            }
            else
            {
                throw ParseError( std::string( "Unknown option: " ) + token );
            }
        }

        // Parse the filenames.
        if ( token.size() == 0 ) throw ParseError( getUsage() );
        options.trainingFile = token;
        if( !( args >> options.outputFile ) ) throw ParseError( getUsage() );

        // Return  results.
        return options;
    }

    std::string  trainingFile;
    std::string  outputFile  ;
    unsigned int maxDepth    ;
    unsigned int treeCount   ;
    unsigned int threadCount ;

  };
}

int main( int argc, char **argv )
{
    try
    {
        // Parse the command-line arguments.
        Options options = Options::parseOptions( argc, argv );

        std::cout <<  options.trainingFile << std::endl;
        std::cout <<  options.outputFile   << std::endl;
        std::cout <<  options.maxDepth     << std::endl;
        std::cout <<  options.treeCount    << std::endl;
        std::cout <<  options.threadCount  << std::endl;


        // Load training data set.
        StopWatch watch;
        std::cout << "Ingesting data..." << std::endl;
        watch.start();
        auto dataSet = loadTrainingDataSet( options.trainingFile );
        std::cout << "Dataset loaded: " << dataSet->size() << " points. (" << watch.stop() << " seconds)." << std::endl;

        // Train a random forest on the data.
        std::cout << "Building indices..." << std::endl;
        watch.start();

        BinaryRandomForestTrainer trainer( options.maxDepth, options.treeCount, options.threadCount );
        std::cout <<"Done (" << watch.stop() << " seconds)." << std::endl;

        std::cout << "Training..." << std::endl;
        watch.start();
        Forest::SharedPointer forest = trainer.train( dataSet );
        std::cout << "Done (" << watch.stop() << " seconds)." << std::endl;

        // Save the model to a file.
        std::cout << "Saving model..." << std::endl;
        watch.start();
        writeToFile( *forest, options.outputFile );
        std::cout << "Done (" << watch.stop() << " seconds)." << std::endl;
    }
    catch ( Exception &e )
    {
        std::cerr << e.getMessage() << std::endl;
        return EXIT_FAILURE;
    }

    // Finish.
    return EXIT_SUCCESS;
}
