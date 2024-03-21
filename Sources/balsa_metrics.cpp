#include <cassert>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>

#include "datatypes.h"
#include "exceptions.h"
#include "serdes.h"
#include "table.h"

namespace
{
class Options
{
public:

    Options()
    {
    }

    static std::string getUsage()
    {
        std::stringstream ss;
        ss << "Usage:" << std::endl
           << std::endl
           << "   balsa_metrics <ground_truth_labels> <classifier_labels>" << std::endl
           << std::endl;
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
            throw ParseError( std::string( "Unknown option: " ) + token );
        }

        // Parse the filenames.
        if ( token.size() == 0 ) throw ParseError( getUsage() );
        options.groundTruthLabelsFile = token;

        if ( !(args >> token) ) throw ParseError( getUsage() );
        options.classifierLabelsFile = token;

        // Return results.
        return options;
    }

  std::string groundTruthLabelsFile;
  std::string classifierLabelsFile;
};

} // namespace

int main( int argc, char ** argv )
{
    try
    {
        // Parse the command-line options.
        auto options = Options::parseOptions( argc, argv );

        // Read the gound truth labels and the classifier labels.
        auto groundTruthLabels = Table<Label>::readFileAs( options.groundTruthLabelsFile );
        auto classifierLabels  = Table<Label>::readFileAs( options.classifierLabelsFile );
        if ( groundTruthLabels.getColumnCount() != 1 ) throw ParseError( "Unexpected number of columns." );
        if ( classifierLabels .getColumnCount() != 1 ) throw ParseError( "Unexpected number of columns." );
        if ( groundTruthLabels.getRowCount() != classifierLabels.getRowCount() ) throw ParseError( "The input files have a different number of points." );

        // Determine the number of classes.
        Label highestClass = 0;
        for ( auto l: groundTruthLabels ) highestClass = std::max( highestClass, l );
        for ( auto l: classifierLabels  ) highestClass = std::max( highestClass, l );

        // Calculate the confusion matrix.
        Table<unsigned int> confusionMatrix( highestClass + 1, highestClass + 1 );
        Table<Label>::ConstIterator classifierIt = classifierLabels.begin();
        for ( auto groundTruth: groundTruthLabels )
        {
            auto classifier = *classifierIt++;
            ++confusionMatrix( classifier, groundTruth );
        }

        // Print the confusion matrix:
        std::cout << confusionMatrix;
    }
    catch ( Exception & e )
    {
        std::cerr << e.getMessage() << std::endl;
        return EXIT_FAILURE;
    }

    // Finish.
    return EXIT_SUCCESS;
}
