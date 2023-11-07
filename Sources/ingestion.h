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

TrainingDataSet loadTrainingDataSet(const std::string &filename)
{
    std::ifstream stream(filename, std::ios::binary);
    assert(stream.good());

    const unsigned int tupleSize = read<std::uint32_t>(stream);
    const unsigned int featureCount = tupleSize - 1;

    std::string formatString(tupleSize, '?');
    stream.read(formatString.data(), formatString.size());
    for (unsigned i = 0; i < featureCount; ++i)
        assert(formatString.at(i) == 'f');
    assert(formatString.back() == 'B');

    TrainingDataSet dataset(featureCount);
    DataPoint point(featureCount);
    while (stream.good())
    {
        for (unsigned int i = 0; i < featureCount; ++i)
        {
            point[i] = read<float>(stream);
        }

        DataPointLabel label = read<DataPointLabel>(stream);

        if (stream.eof())
            break;

        dataset.appendDataPoint(point, label);
    }

    return dataset;
}

#endif