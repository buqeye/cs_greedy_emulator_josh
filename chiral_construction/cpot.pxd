cdef extern from "src/localGt+.h":
  # see https://arxiv.org/pdf/1406.0454.pdf
  double V0(double k, double kk, int pot, int S, int L, int LL, int J, int channel);

  double Vrlocal(double r, int pot, Channel *channel, Lecs *lecs);
  void Vrlocal_affine(double r, int pot, Channel *chan, double *ret);

  ctypedef struct Channel:
    int S
    int L
    int LL
    int J
    int channel

  ctypedef struct Lecs:
    double CS
    double CT
    double C1
    double C2
    double C3
    double C4
    double C5
    double C6
    double C7
    double CNN
    double CPP
