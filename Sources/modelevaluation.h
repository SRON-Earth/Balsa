#ifndef MODELEVALUATION_H
#define MODELEVALUATION_H

#include <random>

#include "table.h"
#include "datatypes.h"
#include "iteratortools.h"

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
    P4 ( numberOfClasses, 1 ),
    ACC( 0                  )
    {
        // Calculate the confusion matrix.
        for ( auto groundTruthIt( groundTruthBegin ); groundTruthIt != groundTruthEnd; ++groundTruthIt )
        {
            auto classifier = *classifierLabels++;
            ++CM( classifier, *groundTruthIt );
        }

        // Calculate the basic counts.
        auto nc = numberOfClasses;
        unsigned int totalTruePositives = 0;
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
                    if ( row == c && col == c )
                    {
                        TP( c, 0 ) = CM( c, c );
                        totalTruePositives += CM( c, c );
                    }

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

        // Calculate the overall accuracy.
        unsigned int totalPoints = std::distance( groundTruthBegin, groundTruthEnd );
        ACC = static_cast<double>( totalTruePositives ) / totalPoints;

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

  /** Confusion Matrix. */
  Table<unsigned int> CM ;

  /** Positives (ground truth) per class. */
  Table<unsigned int> P  ;

  /** Negatives (ground truth) per class. */
  Table<unsigned int> N  ;

  /** True Positives (correct positive predictions) per class. */
   Table<unsigned int> TP ;

  /** True Negatives (correct negative predictions) per class. */
   Table<unsigned int> TN ;

  /** False Positives (incorrect positive predictions) per class. */
   Table<unsigned int> FP ;

  /** False Negatives (incorrect positive predictions) per class. */
   Table<unsigned int> FN ;

  /** Predicted Positives per class. */
   Table<unsigned int> PP ;

  /** Predicted Negatives per class. */
   Table<unsigned int> PN ;

  /** True Positive Rate per class. */
   Table<double>       TPR;

  /** True Negative Rate per class. */
   Table<double>       TNR;

  /** False Positive Rate per class. */
   Table<double>       FPR;

  /** True Negative Rate per class. */
   Table<double>       FNR;

  /** Positive Predictive value per class. */
   Table<double>       PPV;

  /** Negative Predictive value per class. */
   Table<double>       NPV;

  /** F1 score per class. */
   Table<double>       F1 ;

  /** Positive Likelihood Ratio (LR+) per clsas. */
   Table<double>       LRP;

  /** Negative Likelihood Ratio (LR-) per clsas. */
   Table<double>       LRN;

  /** Diagnostic Odds Ratio per class. */
   Table<double>       DOR;

  /** P4 Metric per class. */
   Table<double>       P4 ;

  /** Accuracy (overall). */
  double                    ACC;
};

/**
 * Feature importances of a classifier, based on several different metrics.
 */
class FeatureImportances
{
public:

    /**
     * Constructor. Performs feature importance analysis of a classifier on a dataset.
     */
    template <typename Classifier, typename PointIterator, typename LabelIterator>
    FeatureImportances( const Classifier & classifier, PointIterator pointsBegin, PointIterator pointsEnd, LabelIterator labelBegin, unsigned int featureCount, unsigned int repetitions = 5 ):
    m_accImportance( featureCount, 0 )
    {
        // Determine the feature type from the point iterator type.
        typedef std::remove_cv_t<typename iterator_value_type<PointIterator>::type> FeatureType;
        static_assert( std::is_arithmetic<FeatureType>::value, "Feature type should be an integral or floating point type." );

        // Check preconditions.
        assert( repetitions > 0 );

        // Create a linear table of point indices. This will be shuffled repeatedly later.
        std::size_t              pointCount = std::distance( pointsBegin, pointsEnd ) / featureCount;
        std::vector<std::size_t> shuffling( pointCount, 0 );
        std::iota( shuffling.begin(), shuffling.end(), 0 );

        // Create a random device for shuffling points.
        std::random_device rd;
        std::mt19937       noise( rd() );

        // Calculate a reference score on the original data.
        Table<Label> predictions( pointCount, 1 );
        classifier.classify( pointsBegin, pointsEnd, predictions.begin() );
        ModelStatistics referenceStats( labelBegin, labelBegin + pointCount, predictions.begin(), classifier.getClassCount() );

        // Test the predictive performance when the shuffling is applied separately to each feature.
        for ( unsigned int featureToShuffle = 0; featureToShuffle < featureCount; ++featureToShuffle )
        {
            // Perform several repetitions of the feature shuffling experiment.
            for ( unsigned int i = 0; i < repetitions; ++i )
            {
                // Shuffle the point indices.
                std::shuffle( shuffling.begin(), shuffling.end(), noise );

                // Create a copy of the data and apply the shuffling to the feature under consideration.
                Table<FeatureType> shuffledPoints( 0, featureCount );
                shuffledPoints.reserveRows( pointCount );
                shuffledPoints.append( pointsBegin, pointsEnd );
                for ( std::size_t pointID = 0; pointID < pointCount; ++pointID )
                {
                    shuffledPoints( pointID, featureToShuffle ) = pointsBegin[shuffling[pointID] * featureCount + featureToShuffle];
                }

                // Apply the classifier to the shuffled data.
                Table<Label> shuffledPredictions( pointCount, 1 );
                classifier.classify( shuffledPoints.begin(), shuffledPoints.end(), shuffledPredictions.begin() );

                // Calculate the performance statistics of the model on the shuffled data.
                ModelStatistics shuffledStats( labelBegin, labelBegin + pointCount, shuffledPredictions.begin(), classifier.getClassCount() );

                // Add the stats to the importance totals.
                m_accImportance[featureToShuffle] += shuffledStats.ACC;
            }

            // Calculate the final importance scores.
            m_accImportance[featureToShuffle] = referenceStats.ACC - ( m_accImportance[featureToShuffle] / repetitions );
        }
    }

  double getAccuracyImportance( unsigned int featureID ) const
  {
      assert( featureID < m_accImportance.size() );
      return m_accImportance[featureID];
  }

  std::size_t getFeatureCount() const
  {
      return m_accImportance.size();
  }

private:

  std::vector<double> m_accImportance;
};

} // Namespace 'balsa'.

/**
 * Print the statistics to a stream in human-readable form.
 */
std::ostream &operator<<( std::ostream &out, const balsa::ModelStatistics &stats );

/**
 * Print the feature importances to a stream in human-readable form.
 */
std::ostream &operator<<( std::ostream &out, const balsa::FeatureImportances  &stats );

#endif // MODELEVALUATION_H
