#ifndef EVC_PHASESHIFT_H
#define EVC_PHASESHIFT_H

#include "localGt+.h"


int emulator_startSession (int numLecSets, Lecs *lecSets, double energy, int l);

int emulator_emulate(Lecs *lecSets, double *phaseShift);

int emulator_closeSession();

#endif //EVC_PHASESHIFT_H
