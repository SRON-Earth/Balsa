#ifndef CLASSIFIERSTREAM_H
#define CLASSIFIERSTREAM_H

#include <cassert>

#include "classifier.h"

namespace balsa
{

/**
 * Abstract interface of a class that represents a collection of classifiers
 * that can be iterated.
 */
class ClassifierInputStream
{
public:

    /**
     * Destructor.
     */
    virtual ~ClassifierInputStream()
    {
    }

    /**
     * Return the number of classes distinguished by the classifiers in this
     * stream.
     */
    virtual unsigned int getClassCount() const = 0;

    /**
     * Return the number of features expected by the classifiers in this
     * stream.
     */
    virtual unsigned int getFeatureCount() const = 0;

    /**
     * Rewind the stream to the beginning.
     */
    virtual void rewind() = 0;

    /**
     * Return the next classifier in the stream, or an empty shared pointer when
     * the end of the stream has been reached.
     */
    virtual Classifier::SharedPointer next() = 0;
};

/**
 * Abstract interface of a class that can consume a series classifiers.
 */
class ClassifierOutputStream
{
public:

    /**
     * Constructs an open stream.
     */
    ClassifierOutputStream():
    m_isClosed( false )
    {
    }

    virtual ~ClassifierOutputStream()
    {
        // TODO: Shouldn't call virtual function from the destructor!
        close();
    }

    /**
     * Write a classifier to the output stream.
     * \pre isOpen()
     */
    void write( const Classifier & classifier )
    {
        assert( isOpen() );
        onWrite( classifier );
    }

    void close()
    {
        // Close the stream and let the subclass perform closing actions.
        if ( m_isClosed ) return;
        onClose();
        m_isClosed = true;
    }

    bool isOpen() const
    {
        return !m_isClosed;
    }

private:

    /**
     * Perform subclass-specific operations when the stream is closed.
     */
    virtual void onClose()
    {
    }

    /**
     * Perform the actual write in a subclass-specific way.
     * This is guaranteed to be called only when the stream is still open.
     */
    virtual void onWrite( const Classifier & classifier ) = 0;

    bool m_isClosed;
};

} // namespace balsa

#endif // CLASSIFIERSTREAM_H
