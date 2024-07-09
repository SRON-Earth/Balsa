#include "modelevaluation.h"

namespace
{
template <typename T>
void printClassMetric( std::ostream & out, const std::string & name, const balsa::Table<T> & metric, unsigned int precision = 8 )
{
    out << name << ":";
    for ( auto v : metric ) out << ' ' << std::setw( precision + 4 ) << std::setprecision( precision ) << v;
    out << std::endl;
}

} // Anonymous namespace.

std::ostream & operator<<( std::ostream & out, const balsa::ModelStatistics & stats )
{
    // Print the metrics.
    out << "Confusion Matrix:" << std::endl;
    out << stats.CM << std::endl;

    out << "Counts per class:" << std::endl;
    printClassMetric( out, "P  ", stats.P );
    printClassMetric( out, "N  ", stats.N );
    printClassMetric( out, "PP ", stats.PP );
    printClassMetric( out, "PN ", stats.PN );
    printClassMetric( out, "TP ", stats.TP );
    printClassMetric( out, "TN ", stats.TN );
    printClassMetric( out, "FP ", stats.FP );
    printClassMetric( out, "FN ", stats.FN );
    out << std::endl;

    out << "Global metrics:" << std::endl;
    out << "ACC: " << stats.ACC << std::endl
        << std::endl;

    out << "Metrics per class:" << std::endl;
    printClassMetric( out, "TPR", stats.TPR );
    printClassMetric( out, "TNR", stats.TNR );
    printClassMetric( out, "FPR", stats.FPR );
    printClassMetric( out, "FNR", stats.FNR );
    printClassMetric( out, "PPV", stats.PPV );
    printClassMetric( out, "NPV", stats.NPV );
    printClassMetric( out, "LR+", stats.LRP );
    printClassMetric( out, "LR-", stats.LRN );
    printClassMetric( out, "F1 ", stats.F1 );
    printClassMetric( out, "DOR", stats.DOR );
    printClassMetric( out, "P4 ", stats.P4 );

    return out;
}

std::ostream & operator<<( std::ostream & out, const balsa::FeatureImportances & stats )
{
    out << "-----------------------------------" << std::endl;
    out << "Feature #: Importance (ACC-based): " << std::endl;
    out << "-----------------------------------" << std::endl;
    for ( unsigned int i = 0; i < stats.getFeatureCount(); ++i )
    {
        std::cout << std::setw( 10 ) << std::left << i << " " << stats.getAccuracyImportance( i ) << std::endl;
    }

    return out;
}
