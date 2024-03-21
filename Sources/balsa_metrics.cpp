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
        std::size_t numberOfClasses = highestClass + 1;

        // N.B. Variable names for the metrics follow the naming conventions of the Balsa documentation.

        // Calculate the confusion matrix.
        Table<unsigned int> CM( highestClass + 1, numberOfClasses );
        Table<Label>::ConstIterator classifierIt = classifierLabels.begin();
        for ( auto groundTruth: groundTruthLabels )
        {
            auto classifier = *classifierIt++;
            ++CM( classifier, groundTruth );
        }

        // Calculate the basic metrics per class.
        auto nc = numberOfClasses;
        Table<unsigned int> P( nc, 1 ), N( nc, 1), PP( nc, 1 ), PN( nc, 1 ), TP( nc, 1 ), TN( nc, 1 ), FP( nc, 1 ), FN( nc, 1 );
        for ( Label c = 0; c < nc; ++c )
        {
            // Other metrics.
            for ( Label row = 0; row < nc; ++row )
            {
                for( Label col = 0; col < numberOfClasses; ++col )
                {
                    // Positives.
                    if ( col == c ) P(c,0) += CM(row,col);

                    // Predicted Positives.
                    if ( row == c ) PP(col,0) += CM(row,col);

                    // Negatives.
                    if ( col != c ) N(c, 0) += CM(row,col);

                    // Predicted negatives.
                    if ( row != c ) PN(c, 0) += CM(row,col);

                    // True Postives.
                    if ( row == c && col == c ) TP(c,0) = CM(c,c);

                    // True negatives.
                    if ( row != c && col != c ) TN(c,0) += CM(row,col);

                    // False negatives.
                    if ( row != c && col == c ) FN(c,0) += CM(row,col);

                    // False positives.
                    if ( row == c && col != c ) FN(c,0) += CM(row,col);
                }
            }
        }

        // Calculate the basic metrics.
        Table<double> TPR( numberOfClasses, 1 );
        Table<double> TNR( numberOfClasses, 1 );
        for ( Label l = 0; l < numberOfClasses; ++l )
        {
            TPR(l,0) = static_cast<double>( TP(l,0) ) / P( l, 0 );
            TNR(l,0) = static_cast<double>( TN(l,0) ) / N( l, 0 );
        }

        // Print the metrics.
        std::cout << "Confusion Matrix:" << std::endl;
        std::cout << CM << std::endl;
        std::cout << "P[class]:" << std::endl;
        std::cout << P << std::endl;
        std::cout << "N[class]:" << std::endl;
        std::cout << N << std::endl;
        std::cout << "PP[class]):" << std::endl;
        std::cout << PP << std::endl;
        std::cout << "PN[class]):" << std::endl;
        std::cout << PN << std::endl;
        std::cout << "TP[class]):" << std::endl;
        std::cout << TP << std::endl;
        std::cout << "TN[class]):" << std::endl;
        std::cout << TN << std::endl;
        std::cout << "FP[class]):" << std::endl;
        std::cout << FP << std::endl;
        std::cout << "FN[class]):" << std::endl;
        std::cout << FN << std::endl;
        std::cout << "TPR[class]):" << std::endl;
        std::cout << TPR << std::endl;
        std::cout << "TNR[class]):" << std::endl;
        std::cout << TNR << std::endl;
    }
    catch ( Exception & e )
    {
        std::cerr << e.getMessage() << std::endl;
        return EXIT_FAILURE;
    }

    // Finish.
    return EXIT_SUCCESS;
}
