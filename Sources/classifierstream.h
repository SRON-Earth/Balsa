#ifndef CLASSIFIERSTREAM_H
#define CLASSIFIERSTREAM_H

#include "classifier.h"

namespace balsa
{

/**
 * Abstract interface of a class that represents a collection of classifiers
 * that can be iterated.
 */
template <typename FeatureIterator, typename OutputIterator>
class ClassifierStream
{
public:

    typedef Classifier<FeatureIterator, OutputIterator> ClassifierType;

    /**
     * Destructor.
     */
    virtual ~ClassifierStream()
    {
    }

    /**
     * Return the number of classes distinguished by the classifiers in this
     * stream.
     */
    virtual unsigned int getClassCount() const = 0;

    /**
     * Rewind the stream to the beginning.
     */
    virtual void rewind() = 0;

    /**
     * Return the next classifier in the stream, or an empty shared pointer when
     * the end of the stream has been reached.
     */
    virtual typename ClassifierType::SharedPointer next() = 0;
};

} // namespace balsa

#endif // CLASSIFIERSTREAM_H
