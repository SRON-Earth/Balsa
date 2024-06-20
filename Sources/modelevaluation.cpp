#include "modelevaluation.h"


namespace
{
  template <typename T>
  void printClassMetric( std::ostream &out, const std::string & name, const balsa::Table<T> & metric, unsigned int precision = 8 )
  {
      std::cout << name << ":";
      for ( auto v : metric ) std::cout << ' ' << std::setw( precision + 4 ) << std::setprecision( precision ) << v;
      std::cout << std::endl;
  }

} // Anonymous namespace.

std::ostream &operator<<( std::ostream &out, const balsa::ModelStatistics &stats )
{
    // Print the metrics.
    out << "Confusion Matrix:" << std::endl;
    out << stats.CM << std::endl;

    out << "Counts per class:" << std::endl;
    printClassMetric( out, "P  ", stats.P  );
    printClassMetric( out, "N  ", stats.N  );
    printClassMetric( out, "PP ", stats.PP );
    printClassMetric( out, "PN ", stats.PN );
    printClassMetric( out, "TP ", stats.TP );
    printClassMetric( out, "TN ", stats.TN );
    printClassMetric( out, "FP ", stats.FP );
    printClassMetric( out, "FN ", stats.FN );
    out << std::endl;

    out << "Metrics per class:" << std::endl;
    printClassMetric( out, "TPR", stats.TPR );
    printClassMetric( out, "TNR", stats.TNR );
    printClassMetric( out, "FPR", stats.FPR );
    printClassMetric( out, "FNR", stats.FNR );
    printClassMetric( out, "PPV", stats.PPV );
    printClassMetric( out, "NPV", stats.NPV );
    printClassMetric( out, "LR+", stats.LRP );
    printClassMetric( out, "LR-", stats.LRN );
    printClassMetric( out, "F1 ", stats.F1  );
    printClassMetric( out, "DOR", stats.DOR );
    printClassMetric( out, "P4 ", stats.P4  );
    out << std::endl;

    return out;
}
