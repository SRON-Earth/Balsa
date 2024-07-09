#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "classifierfilestream.h"
#include "config.h"
#include "datatypes.h"
#include "exceptions.h"
#include "fileio.h"
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
           << "   balsa_merge <outfile-name> <balsa-model-file>+" << std::endl;
        return ss.str();
    }

    static Options parseOptions( int argc, char ** argv )
    {
        // Put all arguments in a stringstream.
        std::stringstream args;
        for ( int i = 0; i < argc; ++i )
        {
            args << ' ' << argv[i];
        }

        // Discard the executable name.
        std::string token;
        args >> token;
        token = "";

        // Parse all flags.
        Options options;
        token = "";
        while ( args >> token )
        {
            // Stop if the token is not a flag.
            assert( token.size() );
            if ( token[0] != '-' ) break;
            throw ParseError( std::string( "Unknown option: " ) + token );
        }

        // Parse the output file name,
        if ( token.size() == 0 ) throw ParseError( getUsage() );
        options.outputFile = token;

        // Parse the input filenames.
        while ( args >> token )
        {
            options.modelFiles.push_back( token );
        }
        if ( options.modelFiles.size() < 1 ) throw ParseError( "No input files specified." );

        // Return results.
        return options;
    }

    std::string              outputFile;
    std::vector<std::string> modelFiles;
};

} // namespace

int main( int argc, char ** argv )
{
    try
    {
        // Parse the command-line options.
        auto options = Options::parseOptions( argc, argv );

        // Create the output file.
        EnsembleFileOutputStream out( options.outputFile, "balsa_merge", balsa_VERSION_MAJOR, balsa_VERSION_MINOR, balsa_VERSION_PATCH );

        // Append all input models to the merged file.
        unsigned int classCount   = 0;
        unsigned int featureCount = 0;
        for ( auto & modelFile : options.modelFiles )
        {
            // Open the input file and make sure the model is compatible with what was merged earlier.
            ClassifierFileInputStream in( modelFile );
            if ( classCount != 0 && in.getClassCount() != classCount )
                throw ClientError( "The class count of the model '" + modelFile + "' differs from the earlier input files." );
            if ( featureCount != 0 && in.getFeatureCount() != featureCount )
                throw ClientError( "The feature count of the model '" + modelFile + "' differs from the earlier input files." );
            classCount   = in.getClassCount();
            featureCount = in.getFeatureCount();

            // Append all submodels to the output file.
            while ( auto submodel = in.next() ) out.write( *submodel );
        }

        // Close the merged file.
        out.close();
    }
    catch ( Exception & e )
    {
        std::cerr << e.getMessage() << std::endl;
        return EXIT_FAILURE;
    }

    // Finish.
    return EXIT_SUCCESS;
}
