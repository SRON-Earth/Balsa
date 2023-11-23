#include <iostream>

#include "timing.h"
#include "exceptions.h"
#include "decisiontrees.h"

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

  void writeLabels(const std::vector<bool> &labels, const std::string &filename)
  {
    // Open the output file stream.
    std::ofstream stream( filename.c_str(), std::ofstream::binary );
    assert(stream.good());

    // Write the number of columns.
    const std::uint32_t numColumns = 1;
    stream.write( reinterpret_cast<const char*>( &numColumns ), sizeof( std::uint32_t ) );

    // Write the label values.
    for ( float label : labels )
    {
        stream.write( reinterpret_cast<const char*>( &label ), sizeof( float ) );
    }
  }
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

        // Print model info.
        std::cout << "Model Statistics:" << std::endl
                  << "Maximum Node Count: " << forest.getMaximumNodeCount() << std::endl
                  << "Maximum Depth     : " << forest.getMaximumDepth    () << std::endl;

        // Load the data.
        std::cout << "Ingesting data..." << std::endl;
        watch.start();
        auto dataSet = loadDataSet( options.dataPointFile );
        std::cout << "Dataset loaded: " << dataSet->size() << " points. (" << watch.stop() << " seconds)." << std::endl;

        // Classify the data points.
        std::vector<bool> labels = forest.classify( *dataSet );

        // Store the labels.
        writeLabels( labels, options.outputFile );
    }
    catch ( Exception &e )
    {
        std::cerr << e.getMessage() << std::endl;
        return EXIT_FAILURE;
    }

    // Finish.
    return EXIT_SUCCESS;
}
