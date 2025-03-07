#include "phaseShift.h"
#include "Emulator.h"


Emulator *emu;

int emulator_startSession (int numLecSets, Lecs *lecSets, double energy, int l){
    std::vector<Lecs> trainingPoints(lecSets, lecSets + numLecSets);
    emu = new Emulator{trainingPoints, energy, l};
    return emu == nullptr;
}

int emulator_emulate(Lecs *lecs, double *phaseShift){
    emu->emulate(phaseShift, *lecs);
    return EXIT_SUCCESS;
}

int emulator_closeSession(){
    delete emu;
    return EXIT_SUCCESS;
}