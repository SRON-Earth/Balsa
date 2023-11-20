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
        return  "Usage: rftrain <training input file> <model output file>";
    }

    static Options parseOptions( int argc, char **argv )
    {
        // Check the arguments.
        if ( argc != 3 )
        {
            throw ParseError( getUsage() );
        }

        // Parse the arguments.
        Options options;
        options.trainingFile = std::string( argv[1] );
        options.outputFile   = argv[2];

        return options;
    }

    std::string trainingFile;
    std::string outputFile  ;

  };
}

int main( int argc, char **argv )
{
    try
    {
        // Parse the command-line arguments.
        Options options = Options::parseOptions( argc, argv );

        // Load training data set.
        StopWatch watch;
        std::cout << "Ingesting data..." << std::endl;
        watch.start();
        auto dataSet = loadTrainingDataSet( options.trainingFile );
        std::cout << "Dataset loaded: " << dataSet->size() << " points. (" << watch.stop() << " seconds)." << std::endl;

        // Train a random forest on the data.
        std::cout << "Building indices..." << std::endl;
        unsigned int MAX_DEPTH = 50;
        watch.start();
        BinaryRandomForestTrainer trainer( dataSet, MAX_DEPTH );
        std::cout <<"Done (" << watch.stop() << " seconds)." << std::endl;

        std::cout << "Training..." << std::endl;
        watch.start();
        trainer.train();
        std::cout << "Done (" << watch.stop() << " seconds)." << std::endl;

        // Save the model to a file.
        trainer.saveModel( options.outputFile );

    }
    catch ( Exception &e )
    {
        std::cerr << e.getMessage() << std::endl;
        return EXIT_FAILURE;
    }

    // Finish.
    return EXIT_SUCCESS;
}
