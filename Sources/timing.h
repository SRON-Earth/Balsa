#ifndef TIMING_H
#define TIMING_H

#include <chrono>

/**
 * A class for performing basic timing measurements.
 */
class StopWatch
{
  public:

    typedef std::chrono::time_point<std::chrono::system_clock> Timestamp;
    typedef double Seconds;

    StopWatch():
    m_running( false ),
    m_start( std::chrono::system_clock::now() ),
    m_end  ( std::chrono::system_clock::now() )
    {
    }

    /**
     * Starts counting, returns the current elapsed time.
     */
    Seconds start()
    {
        auto elapsed = getElapsedTime();
        m_running = true;
        m_start = std::chrono::system_clock::now();
        m_end   = m_start;
        return elapsed;
    }

    /**
     * Stops counting, returns the elapsed time.
     */
    Seconds stop()
    {
        if ( m_running )
        {
            m_running = false;
            m_end     = std::chrono::system_clock::now();
        }
        return getElapsedTime();
    }

    /**
     * Returns the time between when the stopwatch was started and when it was stopped, or now, if it is still running.
     */
    Seconds getElapsedTime() const
    {
        auto end = m_running ? std::chrono::system_clock::now() : m_end;
        return std::chrono::duration<double>( end - m_start ).count();
    }

  private:

    bool      m_running;
    Timestamp m_start  ;
    Timestamp m_end    ;
};

#endif // TIMING_H
