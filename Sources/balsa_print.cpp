#include <cassert>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>

#include "datatypes.h"
#include "exceptions.h"
#include "serdes.h"
#include "table.h"

using namespace balsa;

namespace
{
class Options
{
public:

    Options()
    {
    }

    static std::string getUsage()
    {
        std::stringstream ss;
        ss << "Usage:" << std::endl
           << std::endl
           << "   balsa_print <balsa_file>" << std::endl
           << std::endl;
        return ss.str();
    }

    static Options parseOptions( int argc, char ** argv )
    {
        // Put all arguments in a stringstream.
        std::stringstream args;
        for ( int i = 0; i < argc; ++i ) args << ' ' << argv[i];

        // Discard the executable name.
        std::string token;
        args >> token;

        // Parse all flags.
        Options options;
        while ( args >> token )
        {
            // Stop if the token is not a flag.
            assert( token.size() );
            if ( token[0] != '-' ) break;
            throw ParseError( std::string( "Unknown option: " ) + token );
        }

        // Parse the filename.
        if ( token.size() == 0 ) throw ParseError( getUsage() );
        options.fileName = token;

        // Return results.
        return options;
    }

    std::string fileName;
};

} // namespace

int main( int argc, char ** argv )
{
    try
    {
        // Parse the command-line options.
        auto options = Options::parseOptions( argc, argv );

        // Open the input file.
        std::ifstream in;
        in.open( options.fileName, std::ios::binary );

        // Read and print data objects until the end of the file.
        while ( in.peek() != EOF )
        {
            // Peek at the first token.
            auto header = peekFixedSizeToken( in, 4 );

            // Handle the various supported file types.
            if ( header == "frst" )
            {
                // A forest is just a list of trees. Consume the marker and continue.
                expect( in, "frst", "Missing 'frst' header." );
                std::cout << "FOREST" << std::endl;
                continue;
            }
            else if ( header == "tree" )
            {
                // Print the tree info.
                expect( in, "tree", "Missing 'tree' header." );
                expect( in, "fcnt", "Missing feature count marker." );
                auto featureCount = deserialize<uint32_t>( in );
                std::cout << "TREE " << featureCount << " features." << std::endl;

                // Parse the tables that describe the tree.
                Table<NodeID>    left( 1 ), right( 1 );
                Table<FeatureID> featureID( 1 );
                Table<double>    featureValue( 1 );
                Table<Label>     label( 1 );
                in >> left;
                in >> right;
                in >> featureID;
                in >> featureValue;
                in >> label;

                // Print the values.
                std::cout << "N:   L:   R:   F:   V:              L:" << std::endl;
                for ( unsigned int row = 0; row < left.getRowCount(); ++row )
                {
                    std::cout << std::left << std::setw( 4 ) << row << " "
                              << std::left << std::setw( 4 ) << left( row, 0 ) << " " << std::setw( 4 ) << right( row, 0 ) << " "
                              << std::left << std::setw( 4 ) << static_cast<int>( featureID( row, 0 ) ) << " " << std::setw( 4 ) << std::setw( 16 ) << featureValue( row, 0 )
                              << std::left << std::setw( 4 ) << int( label( row, 0 ) ) << std::endl;
                    ;
                }
            }
            else if ( header == "tabl" )
            {
                // Peek at the table specification.
                std::size_t rowCount( 0 ), columnCount( 0 );
                std::string typeName;
                auto        position = in.tellg();
                Table<uint32_t>::parseTableSpecification( in, rowCount, columnCount, typeName );
                in.seekg( position );

                // Parse and print the table.
                if ( typeName == getTypeName<Label>() )
                {
                    Table<Label> t( 1 );
                    in >> t;
                    std::cout << t;
                }
                else if ( typeName == getTypeName<float>() )
                {
                    Table<float> t( 1 );
                    in >> t;
                    std::cout << t;
                }
                else if ( typeName == getTypeName<double>() )
                {
                    Table<double> t( 1 );
                    in >> t;
                    std::cout << t;
                }
            }
        }
    }
    catch ( Exception & e )
    {
        std::cerr << e.getMessage() << std::endl;
        return EXIT_FAILURE;
    }

    // Finish.
    return EXIT_SUCCESS;
}
