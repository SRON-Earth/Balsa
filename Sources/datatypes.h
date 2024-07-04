#ifndef DATATYPES_H
#define DATATYPES_H

#include <cstdint>
#include "table.h"

namespace balsa
{

/**
 * The integer type used to identify a feature dimension in a data point or data set.
 */
typedef uint8_t FeatureID;

/**
 * The integer type used as a label type in classification problems.
 */
typedef uint8_t Label;

/**
 * The integer type used to identify one point in a data set.
 */
typedef uint32_t DataPointID;

/**
 * The integer type used to identify a node in a decision tree.
 */
typedef uint32_t NodeID;

/**
 * A table of votes for voted classification. The index is a Label, the value a vote count.
 */
typedef Table<uint32_t> VoteTable;

} // namespace balsa

#endif // DATATYPES_H
