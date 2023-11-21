#ifndef FOO_H
#define FOO_H

#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>

#include "datamodel.h"

template <typename T>
T read(std::istream &stream)
{
    T result;
    std::array<char, sizeof(T)> buffer;
    stream.read(buffer.data(), buffer.size());
    std::memcpy(&result, buffer.data(), buffer.size());
    return result;
}

TrainingDataSet::SharedPointer loadTrainingDataSet(const std::string &filename)
{
    // Open the file stream.
    std::ifstream stream(filename, std::ios::binary);
    assert(stream.good());

    // Read the column count of the input matrix.
    const unsigned int numColumns = read<std::uint32_t>(stream);
    const unsigned int numFeatures = numColumns - 1;

    // Read the rows of points.
    TrainingDataSet::SharedPointer dataset(new TrainingDataSet(numFeatures));
    while (stream.good())
    {
        // Read the data point columns.
        DataPoint point(numFeatures);
        for (unsigned int i = 0; i < numFeatures; ++i)
            point.at(i) = read<float>(stream);

        // Read the label.
        const bool label = (read<float>(stream) != 0.0f);

        // Stop if the end of the input file was reached.
        if (stream.eof())
            break;

        // Add the point to the dataset.
        dataset->appendDataPoint(point, label);
    }

    return dataset;
}

#endif
