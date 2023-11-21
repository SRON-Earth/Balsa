#ifndef FOO_H
#define FOO_H

#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>

#include "datamodel.h"

namespace
{
template <typename T>
T read(std::istream &stream)
{
    T result;
    std::array<char, sizeof(T)> buffer;
    stream.read(buffer.data(), buffer.size());
    std::memcpy(&result, buffer.data(), buffer.size());
    return result;
}
} // Anonymous namespace

DataSet::SharedPointer loadDataSet(const std::string &dataFile)
{
    // Open the file stream.
    std::ifstream stream(dataFile, std::ios::binary);
    assert(stream.good());

    // Read the column count of the input matrix.
    const unsigned int numColumns = read<std::uint32_t>(stream);

    // Read the rows of points.
    DataSet::SharedPointer dataset(new DataSet(numColumns));
    while (stream.good())
    {
        // Read the data point columns.
        DataPoint point(numColumns);
        for (unsigned int i = 0; i < numColumns; ++i)
            point.at(i) = read<float>(stream);

        // Stop if the end of the input file was reached.
        if (stream.eof())
            break;

        // Add the point to the dataset.
        dataset->appendDataPoint(point);
    }

    return dataset;
}

TrainingDataSet::SharedPointer loadTrainingDataSet(const std::string &dataFile, const std::string &labelFile)
{
    // Open the data file stream.
    std::ifstream dataStream(dataFile, std::ios::binary);
    assert(dataStream.good());

    // Open the label file stream.
    std::ifstream labelStream(labelFile, std::ios::binary);
    assert(labelStream.good());

    // Read the column count of the inputs.
    const unsigned int numDataColumns = read<std::uint32_t>(dataStream);
    const unsigned int numLabelColumns = read<std::uint32_t>(labelStream);
    assert(numLabelColumns == 1);

    // Read the rows of points.
    TrainingDataSet::SharedPointer dataset(new TrainingDataSet(numDataColumns));
    while (dataStream.good() && labelStream.good())
    {
        // Read the data point columns.
        DataPoint point(numDataColumns);
        for (unsigned int i = 0; i < numDataColumns; ++i)
            point.at(i) = read<float>(dataStream);

        // Read the label.
        const bool label = (read<float>(labelStream) != 0.0f);

        // Stop if the end of the input file was reached.
        if (dataStream.eof() || labelStream.eof())
        {
            assert(dataStream.eof() && labelStream.eof());
            break;
        }

        // Add the point to the dataset.
        dataset->appendDataPoint(point, label);
    }

    return dataset;
}

#endif
