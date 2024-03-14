#include "weightedcoin.h"

MasterSeedSequence & getMasterSeedSequence()
{
    static MasterSeedSequence seedSequence;
    return seedSequence;
}
