#include <iostream>

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


    static constexpr const char * getUsage()
    {
        return  "Usage: rfclassify <model file> <datapoint file> <output file>";
    }

    static Options parseOptions( int argc, char **argv )
    {
        // Check the arguments.
        if ( argc != 4 )
        {
            throw ParseError( getUsage() );
        }

        // Parse the arguments.
        Options options;
        options.modelFile     = argv[1];
        options.dataPointFile = argv[2];
        options.outputFile    = argv[3];

        return options;
    }

    std::string modelFile    ;
    std::string dataPointFile;
    std::string outputFile   ;

  };
}

int main( int argc, char **argv )
{
    try
    {
        // Parse the command-line arguments.
        Options options = Options::parseOptions( argc, argv );

        // Load the random forest model.
        StopWatch watch;
        std::cout << "Loading model.." << std::endl;
        watch.start();
        Forest forest = loadForest( options.modelFile );
        std::cout << "Done (" << watch.stop() << " seconds)." << std::endl;

        // Load the data.
        //std::cout << "Ingesting data..." << std::endl;
        //auto dataSet = loadTrainingDataSet( options.dataPointFile );
        //std::cout << "Dataset loaded: " << dataSet->size() << " points. (" << elapsed.count() << " seconds)." << std::endl;
    }
    catch ( Exception &e )
    {
        std::cerr << e.getMessage() << std::endl;
        return EXIT_FAILURE;
    }

    // Finish.
    return EXIT_SUCCESS;
}
