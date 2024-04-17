#include "weightedcoin.h"

namespace balsa
{

MasterSeedSequence & getMasterSeedSequence()
{
    static MasterSeedSequence seedSequence;
    return seedSequence;
}

} // namespace balsa
