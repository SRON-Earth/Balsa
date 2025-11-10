#ifndef RANDOMFORESTCLASSIFIER_H
#define RANDOMFORESTCLASSIFIER_H

#include "classifierfilestream.h"
#include "ensembleclassifier.h"

namespace balsa
{
class RandomForestClassifier: public EnsembleClassifier
{
public:

    RandomForestClassifier( const std::string & modelFileName, unsigned int maxThreads = 0, unsigned int maxPreload = 1 ):
    m_classifierStream( modelFileName, maxPreload )
    {
        init( m_classifierStream, maxThreads );
    }

private:

    ClassifierFileInputStream m_classifierStream;
};

} // namespace balsa

#endif // RANDOMFORESTCLASSIFIER_H
