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
        Forest::SharedPointer forest = loadForest( options.modelFile );
        std::cout << "Done." << std::endl;
        const auto modelLoadTime = watch.getElapsedTime();

        // Print model info.
        std::cout << "Model Statistics:" << std::endl
                  << "Maximum Node Count: " << forest->getMaximumNodeCount() << std::endl
                  << "Maximum Depth     : " << forest->getMaximumDepth    () << std::endl;

        // Load the data.
        std::cout << "Ingesting data..." << std::endl;
        watch.start();
        auto dataSet = loadDataSet( options.dataPointFile );
        std::cout << "Dataset loaded: " << dataSet->size() << " points." << std::endl;
        const auto dataLoadTime = watch.getElapsedTime();

        // Classify the data points.
        watch.start();
        std::vector<bool> labels = forest->classify( *dataSet );
        watch.stop();
        const auto classificationTime = watch.getElapsedTime();

        // Store the labels.
        watch.start();
        writeLabels( labels, options.outputFile );
        watch.stop();
        const auto labelStoreTime = watch.getElapsedTime();

        std::cout << "Timings:" << std::endl
                  << "Model Load Time: " << modelLoadTime << std::endl
                  << "Data Load Time: " << dataLoadTime << std::endl
                  << "Classification Time: " << classificationTime << std::endl
                  << "Label Store Time: " << labelStoreTime << std::endl;
    }
    catch ( Exception &e )
    {
        std::cerr << e.getMessage() << std::endl;
        return EXIT_FAILURE;
    }

    // Finish.
    return EXIT_SUCCESS;
}
