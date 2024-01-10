#ifndef MESSAGEQUEUE_H
#define MESSAGEQUEUE_H

#include <queue>
#include <mutex>
#include <condition_variable>

/**
 * A thread-safe queue for distributing messages over threads.
 */
template<typename Message>
class MessageQueue
{
  public:

    /**
     * Append a message to the back of the queue.
     */
    void send( const Message &message )
    {
        // Critical section.
        {
            // Acquire the mutex on the queue.
            std::lock_guard<std::mutex> lock( m_mutex );

            // Add an item to the queue.
            m_queue.push( message );
        }

        // Wake up one waiter to pick up the message.
        m_condition.notify_one();
    }

    /**
     * Remove one message from the front of the queue.
     */
    Message receive()
    {
        // Acquire the mutex on the queue.
        std::unique_lock<std::mutex> lock( m_mutex );

        // Wait for the queue to contain at least one item.
        while( m_queue.empty() ) m_condition.wait(lock);

        // Pop and return the first item.
        auto message = m_queue.front();
        m_queue.pop();
        return message;
    }

  private:

      std::queue<Message>     m_queue    ;
      mutable std::mutex      m_mutex    ;
      std::condition_variable m_condition;

};

#endif // MESSAGEQUEUE_H
