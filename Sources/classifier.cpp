#include "Classifier.h"

namespace balsa
{

ClassifierVisitor::~ClassifierVisitor()
{
}

void ClassifierVisitor::visit( const EnsembleClassifier             &classifier )
{
}

void ClassifierVisitor::visit( const DecisionTreeClassifier<float>  &classifier )
{
}

void ClassifierVisitor::visit( const DecisionTreeClassifier<double> &classifier )
{
}

}
