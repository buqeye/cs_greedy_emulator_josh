#ifndef LOCALGT_POTENTIAL
#define LOCALGT_POTENTIAL

double V0(double k, double kk, int pot, int S, int L, int LL, int J, int channel);

struct Lecs{double CS, CT, C1, C2, C3, C4, C5, C6, C7, CNN, CPP;};
struct Channel {int S; int L; int LL; int J; int channel;};

double Vrlocal(double r, int pot, Channel *channel, Lecs *lecs);
void Vrlocal_affine(double r, int pot, Channel *chan, double *ret);

#endif // LOCALGT_POTENTIAL
