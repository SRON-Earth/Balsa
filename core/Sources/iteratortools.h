#ifndef ITERATORTOOLS_H
#define ITERATORTOOLS_H

#include <iterator>

namespace balsa
{

/**
 * Iterator trait class whose \c type member typedef is equal to the type of the
 * values being iterated. Unlike std::iterator_traits<>, \c type is
 * well-defined also for insert iterators.
 */
template <typename T, typename = void>
struct iterator_value_type
{
    using type = typename std::iterator_traits<T>::value_type;
};

template <typename T>
struct iterator_value_type<T, std::void_t<typename T::container_type>>
{
    using type = typename T::container_type::value_type;
};

} // namespace balsa

#endif // ITERATORTOOLS_H
