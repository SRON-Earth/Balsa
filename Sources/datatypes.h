#ifndef DATATYPES_H
#define DATATYPES_H

#include <cstdint>

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

#endif // DATATYPES_H
