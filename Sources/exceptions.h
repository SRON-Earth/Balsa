#ifndef EXCEPTIONS_H
#define EXCEPTIONS_H

#include <string>

/**
 * Base class of all exceptions.
 */
class Exception
{
public:

  /**
   * Constructor.
   */
  Exception( const std::string &message ):
  m_message( message )
  {
  }

  /**
   * Destructor.
   */
  virtual ~Exception()
  {
  }

  /**
   * Returns the error message.
   */
  std::string getMessage() const
  {
      return m_message;
  }

private:

  std::string m_message;
};

/**
 * An exception thrown when an error occurs that is due to a mistake of the caller (the client) of a function.
 */
class ClientError: public Exception
{
public:

  ClientError( const std::string &message ):
  Exception( message )
  {
  }
};

/**
 * An exception thrown when an error occurs that is due to a problem on the callee-side of a function (the supplier-side).
 */
class SupplierError: public Exception
{
public:

    SupplierError( const std::string &message ):
    Exception( message )
    {
    }
};

/**
 * Thrown when a grammatical error is encountered in a parsed block of data.
 */
class ParseError: public ClientError
{
public:

  ParseError( const std::string &message ):
  ClientError( message )
  {
  }
};

#endif // EXCEPTIONS_H
