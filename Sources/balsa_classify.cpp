#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>

#include "config.h"
#include "datatypes.h"
#include "exceptions.h"
#include "fileio.h"
#include "randomforestclassifier.h"
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
           << "   balsa_classify [options] <model file> [<datapoint file>]+" << std::endl
           << std::endl
           << " Options:" << std::endl
           << std::endl
           << "   -t <thread count>   : Number of threads (default: 1)." << std::endl
           << "   -p <preload count>  : Number of trees to preload (default: 1)." << std::endl
           << "   -cw <label> <weight>: Sets class weight (see below). (default: 1)." << std::endl
           << std::endl
           << "The class/label for each point is determined by counting the votes of a set of" << std::endl
           << "independently trained, randomized decision trees. The user can provide a class" << std::endl
           << "weight to skew the vote of a particular class. The votes in favor of the" << std::endl
           << "class for which the weight is provided will be multiplied with the weight," << std::endl
           << "before the maximum value is determined." << std::endl;
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

            // Parse the '-t <threadcount>' option.
            if ( token == "-t" )
            {
                if ( !( args >> options.threadCount ) ) throw ParseError( "Missing parameter to -t option." );
            }
            else if ( token == "-p" )
            {
                if ( !( args >> options.maxPreload ) ) throw ParseError( "Missing parameter to -p option." );
            }
            else if ( token == "-cw" )
            {
                unsigned int label  = 0;
                float        weight = 0;
                if ( !( args >> label  ) ) throw ParseError( "Missing class parameter to -cw option." );
                if ( !( args >> weight ) ) throw ParseError( "Missing weight parameter to -cw option." );
                options.m_classWeights.push_back( std::tuple<unsigned int, float>( label, weight ) );
            }
            else
            {
                throw ParseError( std::string( "Unknown option: " ) + token );
            }
        }

        // Parse the model file name.
        if ( token.size() == 0 ) throw ParseError( getUsage() );
        options.modelFile = token;

        // Parse the input file names.
        std::string fileName;
        while ( args >> fileName )
        {
            auto path = std::filesystem::path( fileName );
            if ( !path.has_filename() ) throw ParseError( "Input path has no filename." );
            options.dataFiles.push_back( fileName );
        }
        if ( options.dataFiles.size() == 0 ) throw ParseError( "No input files." );

        // Return  results.
        return options;
    }

    std::string                                  modelFile;
    std::vector<std::string>                     dataFiles;
    unsigned int                                 threadCount;
    unsigned int                                 maxPreload;
    std::vector<std::tuple<unsigned int, float>> m_classWeights;
};

std::string createOutputFileName( const std::string &inputFilePath )
{
    // Extract the base name and the extensions.
    auto path =  std::filesystem::path( inputFilePath );

    // Create the output file name.
    auto extension = path.extension();
    if ( extension != ".balsa" ) extension += ".balsa";
    std::string outFile = path.stem().string() + "-predictions" + extension.string();
    return outFile;
}

} // namespace

int main( int argc, char ** argv )
{
    try
    {
        // Parse the command-line arguments.
        Options options = Options::parseOptions( argc, argv );

        // Debug.
        std::cout << "Model File : " << options.modelFile << std::endl;
        std::cout << "Data Files :";
        for ( auto &f: options.dataFiles ) std::cout << ' ' << f << std::endl;
        std::cout << "Threads    : " << options.threadCount << std::endl;
        std::cout << "Preload    : " << options.maxPreload << std::endl;
        std::cout << std::endl;
        assert( options.threadCount > 0 );

        // Create a random forest classifier.
        RandomForestClassifier< Table<double>::ConstIterator, Table<Label>::Iterator> classifier( options.modelFile, options.threadCount - 1, options.maxPreload );

        // Override the class weights.
        std::vector<float> weights( classifier.getClassCount(), 1.0 );
        for ( auto &pair: options.m_classWeights )
        {
            auto label = std::get<0>( pair );
            auto weight = std::get<1>( pair );
            if ( label >= weights.size() ) throw ClientError( "Class out of range: " + std::to_string( label ) );
            if ( weight != weight || weight < 0 ) throw ClientError( "Invalid weight: " + std::to_string( weight ) );
            weights[label] = weight;
        }
        classifier.setClassWeights( weights );

        // Load and classify all files, measuring the duration.
        StopWatch::Seconds dataLoadTime       = 0;
        StopWatch::Seconds classificationTime = 0;
        StopWatch::Seconds labelStoreTime     = 0;
        for ( auto & dataFile: options.dataFiles )
        {
            // Load the data.
            StopWatch watch;
            std::cout << "Ingesting data..." << std::endl;
            watch.start();
            auto dataSet = readTableAs<double>( dataFile );
            std::cout << "Dataset loaded: " << dataSet.getColumnCount() << " features x " << dataSet.getRowCount() << " points." << std::endl;
            dataLoadTime += watch.getElapsedTime();

            // Classify the data.
            watch.start();
            Table<Label> labels( dataSet.getRowCount(), 1 );
            classifier.classify( dataSet.begin(), dataSet.end(), dataSet.getColumnCount(), labels.begin() );
            watch.stop();
            classificationTime += watch.getElapsedTime();

            // Store the labels.
            watch.start();
            BalsaFileWriter fileWriter( createOutputFileName( dataFile ) );
            fileWriter.setCreatorName( "balsa_classify" );
            fileWriter.setCreatorMajorVersion( balsa_VERSION_MAJOR );
            fileWriter.setCreatorMinorVersion( balsa_VERSION_MINOR );
            fileWriter.setCreatorPatchVersion( balsa_VERSION_PATCH );
            fileWriter.writeTable( labels );
            watch.stop();
            labelStoreTime += watch.getElapsedTime();
        }

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
