#ifndef CLASSIFIERVISITOR_H
#define CLASSIFIERVISITOR_H

namespace balsa
{

// Forward declarations of all supported classifiers.
class EnsembleClassifier;
template <typename FeatureType>
class DecisionTreeClassifier;

/**
 * Base class for visiting Classifiers.
 */
class ClassifierVisitor
{
public:

    virtual ~ClassifierVisitor()
    {
    }

    virtual void visit( const EnsembleClassifier & classifier )             = 0;
    virtual void visit( const DecisionTreeClassifier<float> & classifier )  = 0;
    virtual void visit( const DecisionTreeClassifier<double> & classifier ) = 0;
};

} // namespace balsa

#endif // CLASSIFIERVISITOR_H
