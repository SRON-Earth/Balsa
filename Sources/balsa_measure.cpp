#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "datatypes.h"
#include "exceptions.h"
#include "serdes.h"
#include "table.h"

using namespace balsa;

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
           << "   balsa_measure <ground_truth_labels> <classifier_labels>" << std::endl
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

        if ( !( args >> token ) ) throw ParseError( getUsage() );
        options.classifierLabelsFile = token;

        // Return results.
        return options;
    }

    std::string groundTruthLabelsFile;
    std::string classifierLabelsFile;
};

} // namespace

template <typename T>
void printClassMetric( const std::string & name, const Table<T> & metric, unsigned int precision = 8 )
{
    std::cout << name << ":";
    for ( auto v : metric ) std::cout << ' ' << std::setw( precision + 4 ) << std::setprecision( precision ) << v;
    std::cout << std::endl;
}

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
        if ( classifierLabels.getColumnCount() != 1 ) throw ParseError( "Unexpected number of columns." );
        if ( groundTruthLabels.getRowCount() != classifierLabels.getRowCount() ) throw ParseError( "The input files have a different number of points." );

        // Determine the number of classes.
        Label highestClass = 0;
        for ( auto l : groundTruthLabels ) highestClass = std::max( highestClass, l );
        for ( auto l : classifierLabels ) highestClass = std::max( highestClass, l );
        std::size_t numberOfClasses = highestClass + 1;

        // N.B. Variable names for the metrics follow the naming conventions of the Balsa documentation.

        // Calculate the confusion matrix.
        Table<unsigned int>         CM( highestClass + 1, numberOfClasses );
        Table<Label>::ConstIterator classifierIt = classifierLabels.begin();
        for ( auto groundTruth : groundTruthLabels )
        {
            auto classifier = *classifierIt++;
            ++CM( classifier, groundTruth );
        }

        // Calculate the basic counts.
        auto                nc = numberOfClasses;
        Table<unsigned int> P( nc, 1 ), N( nc, 1 ), TP( nc, 1 ), TN( nc, 1 ), FP( nc, 1 ), FN( nc, 1 ), PP( nc, 1 ), PN( nc, 1 );
        for ( Label c = 0; c < nc; ++c )
        {
            // Other metrics.
            for ( Label row = 0; row < nc; ++row )
            {
                for ( Label col = 0; col < numberOfClasses; ++col )
                {
                    // Positives.
                    if ( col == c ) P( col, 0 ) += CM( row, col );

                    // Negatives.
                    if ( col != c ) N( c, 0 ) += CM( row, col );

                    // True Postives.
                    if ( row == c && col == c ) TP( c, 0 ) = CM( c, c );

                    // True negatives.
                    if ( row != c && col != c ) TN( c, 0 ) += CM( row, col );

                    // False negatives.
                    if ( row != c && col == c ) FN( c, 0 ) += CM( row, col );

                    // False positives.
                    if ( row == c && col != c ) FP( c, 0 ) += CM( row, col );
                }
            }

            PP( c, 0 ) = TP( c, 0 ) + FP( c, 0 );
            PN( c, 0 ) = TN( c, 0 ) + FN( c, 0 );
        }

        // Calculate per-class metrics.
        Table<double> TPR( numberOfClasses, 1 );
        Table<double> TNR( numberOfClasses, 1 );
        Table<double> FPR( numberOfClasses, 1 );
        Table<double> FNR( numberOfClasses, 1 );
        Table<double> PPV( numberOfClasses, 1 );
        Table<double> NPV( numberOfClasses, 1 );
        Table<double> F1( numberOfClasses, 1 );
        Table<double> LRP( numberOfClasses, 1 );
        Table<double> LRN( numberOfClasses, 1 );
        Table<double> DOR( numberOfClasses, 1 );
        Table<double> P4( numberOfClasses, 1 );
        for ( Label l = 0; l < numberOfClasses; ++l )
        {
            TPR( l, 0 ) = static_cast<double>( TP( l, 0 ) ) / P( l, 0 );
            TNR( l, 0 ) = static_cast<double>( TN( l, 0 ) ) / N( l, 0 );
            FPR( l, 0 ) = static_cast<double>( FP( l, 0 ) ) / N( l, 0 );
            FNR( l, 0 ) = static_cast<double>( FN( l, 0 ) ) / P( l, 0 );
            PPV( l, 0 ) = static_cast<double>( TP( l, 0 ) ) / PP( l, 0 );
            NPV( l, 0 ) = static_cast<double>( TN( l, 0 ) ) / PN( l, 0 );

            LRP( l, 0 ) = TPR( l, 0 ) / ( 1.0 - TNR( l, 0 ) );
            LRN( l, 0 ) = ( 1.0 - TPR( l, 0 ) ) / TNR( l, 0 );

            F1( l, 0 )  = 2.0 * PPV( l, 0 ) * TPR( l, 0 ) / ( PPV( l, 0 ) + TPR( l, 0 ) );
            DOR( l, 0 ) = LRP( l, 0 ) / LRN( l, 0 );
            P4( l, 0 )  = 4.0 / ( ( 1.0 / TPR( l, 0 ) ) + ( 1.0 / TNR( l, 0 ) ) + ( 1.0 / PPV( l, 0 ) ) + ( 1.0 / NPV( l, 0 ) ) );
        }

        // Print the metrics.
        std::cout << "Confusion Matrix:" << std::endl;
        std::cout << CM << std::endl;

        std::cout << "Counts per class:" << std::endl;
        printClassMetric( "P  ", P );
        printClassMetric( "N  ", N );
        printClassMetric( "PP ", PP );
        printClassMetric( "PN ", PN );
        printClassMetric( "TP ", TP );
        printClassMetric( "TN ", TN );
        printClassMetric( "FP ", FP );
        printClassMetric( "FN ", FN );
        std::cout << std::endl;

        std::cout << "Metrics per class:" << std::endl;
        printClassMetric( "TPR", TPR );
        printClassMetric( "TNR", TNR );
        printClassMetric( "FPR", FPR );
        printClassMetric( "FNR", FNR );
        printClassMetric( "PPV", PPV );
        printClassMetric( "NPV", NPV );
        printClassMetric( "LR+", LRP );
        printClassMetric( "LR-", LRN );
        printClassMetric( "F1 ", F1  );
        printClassMetric( "DOR", DOR );
        printClassMetric( "P4 ", P4  );
        std::cout << std::endl;
    }
    catch ( Exception & e )
    {
        std::cerr << e.getMessage() << std::endl;
        return EXIT_FAILURE;
    }

    // Finish.
    return EXIT_SUCCESS;
}
