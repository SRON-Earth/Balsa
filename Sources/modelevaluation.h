#ifndef MODELEVALUATION_H
#define MODELEVALUATION_H

#include "table.h"
#include "datatypes.h"

namespace balsa
{

class ModelStatistics
{
  public:

    template<typename GroundTruthLabelIterator, typename ClassifierLabelIterator>
    ModelStatistics( GroundTruthLabelIterator groundTruthBegin, GroundTruthLabelIterator groundTruthEnd, ClassifierLabelIterator classifierLabels, std::size_t numberOfClasses ):
    CM ( numberOfClasses, numberOfClasses ),
    P  ( numberOfClasses, 1 ),
    N  ( numberOfClasses, 1 ),
    TP ( numberOfClasses, 1 ),
    TN ( numberOfClasses, 1 ),
    FP ( numberOfClasses, 1 ),
    FN ( numberOfClasses, 1 ),
    PP ( numberOfClasses, 1 ),
    PN ( numberOfClasses, 1 ),
    TPR( numberOfClasses, 1 ),
    TNR( numberOfClasses, 1 ),
    FPR( numberOfClasses, 1 ),
    FNR( numberOfClasses, 1 ),
    PPV( numberOfClasses, 1 ),
    NPV( numberOfClasses, 1 ),
    F1 ( numberOfClasses, 1 ),
    LRP( numberOfClasses, 1 ),
    LRN( numberOfClasses, 1 ),
    DOR( numberOfClasses, 1 ),
    P4 ( numberOfClasses, 1 )
    {
        // Calculate the confusion matrix.
        for ( auto groundTruthIt( groundTruthBegin ); groundTruthIt != groundTruthEnd; ++groundTruthIt )
        {
            auto classifier = *classifierLabels++;
            ++CM( classifier, *groundTruthIt );
        }

        // Calculate the basic counts.
        auto nc = numberOfClasses;
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
        for ( Label l = 0; l < numberOfClasses; ++l )
        {
            TPR( l, 0 ) = static_cast<double>( TP( l, 0 ) ) / P( l, 0 );
            TNR( l, 0 ) = static_cast<double>( TN( l, 0 ) ) / N( l, 0 );
            FPR( l, 0 ) = static_cast<double>( FP( l, 0 ) ) / N( l, 0 );
            FNR( l, 0 ) = static_cast<double>( FN( l, 0 ) ) / P( l, 0 );
            PPV( l, 0 ) = static_cast<double>( TP( l, 0 ) ) / PP( l, 0 );
            NPV( l, 0 ) = static_cast<double>( TN( l, 0 ) ) / PN( l, 0 );

            LRP( l, 0 ) = TPR( l, 0 ) / FPR( l, 0 );
            LRN( l, 0 ) = FNR( l, 0 ) / TNR( l, 0 );

            F1( l, 0 )  = 2.0 * PPV( l, 0 ) * TPR( l, 0 ) / ( PPV( l, 0 ) + TPR( l, 0 ) );
            DOR( l, 0 ) = LRP( l, 0 ) / LRN( l, 0 );
            P4( l, 0 )  = 4.0 / ( ( 1.0 / TPR( l, 0 ) ) + ( 1.0 / TNR( l, 0 ) ) + ( 1.0 / PPV( l, 0 ) ) + ( 1.0 / NPV( l, 0 ) ) );
        }
    }

    // N.B. Variable names for the metrics follow the naming conventions of the Balsa documentation.
    Table<unsigned int> CM ;
    Table<unsigned int> P  ;
    Table<unsigned int> N  ;
    Table<unsigned int> TP ;
    Table<unsigned int> TN ;
    Table<unsigned int> FP ;
    Table<unsigned int> FN ;
    Table<unsigned int> PP ;
    Table<unsigned int> PN ;
    Table<double>       TPR;
    Table<double>       TNR;
    Table<double>       FPR;
    Table<double>       FNR;
    Table<double>       PPV;
    Table<double>       NPV;
    Table<double>       F1 ;
    Table<double>       LRP;
    Table<double>       LRN;
    Table<double>       DOR;
    Table<double>       P4 ;
};

} // Namespace 'balsa'.


/**
 * Print the statistics to a stream in human-readable form.
 */
std::ostream &operator<<( std::ostream &out, const balsa::ModelStatistics &stats );

#endif // MODELEVALUATION_H
