/*************************************************************************
**************************************************************************
**                                                                      **
**      Local chiral potential: C-Routine by Ingo Tews,                 **
**                              updated Dec 23, 2016                    **
**                                                                      **
**  This routine is written to give partial wave matrix elements for    **
**  the local chiral interactions at LO, NLO, and N2LO in momentum      **
**  space. It defines a function                                        **
**                                                                      **
**              V0(k, kk, pot, S, L, LL, J, channel).                   **
**                                                                      **
**  The input momenta k and kk are of type double and have to have      **
**  the unit fm^-1.                                                     **
**                                                                      **
**  The integer 'pot' choses the potential, cutoff, and SFR cutoff:     **
**                                                                      **
**  pot: xyz: 	x-> order:  LO=0, NLO=1, N2LO=2,                        **
**              y-> cutoff: 0.8=0, 1.0=1, 1.2=2, 0.9=3, 1.1=4,          **
**              z-> SFR cutoff: 800=2, 1000=3, 1200=4, 1400=5.          **
**                                                                      **
**  The integers S, L, LL, and J define the partial wave. Finally,      **
**  the integer 'channel' chooses between nn, np, and pp:               **
**                                                                      **
**  Channel: nn -> -1, np -> 0, pp -> 1.                                **
**                                                                      **
**  The potential is calculated in MeV^-2.                              **
**                                                                      **
**  The potentials includes CIB and CSB effects already at LO. For      **
**  more details see                                                    **
**  Gezerlis et al., Phys.Rev. C90 (2014), 054323.                      **
**                                                                      **
**  If you find any mistakes, please send and email to                  **
**  itews@uw.edu                                                        **
**                                                                      **
**************************************************************************
*************************************************************************/

#define DOCONT  1
#define DOPION  1
#define DOTWOPION  1

#include <iostream>
#include <cstring>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_math.h>
#include <math.h>
#include <localGt+.h>

using namespace std;

static bool output_Vr = false;
static double pi = 2.0*acos(0.0);
static double hbarc = 197.327; // 1 = hbarc MeV fm
static double Mneu = 939.56563;  // neutron mass in MeV
static double Mpro = 938.27205;  // proton mass in MeV
static double Mn = 0.5 * (Mneu + Mpro);

static char interaction[80];

static int unknown_potential = 0;

enum LECN{iVconst, iCS, iCT, iC1, iC2, iC3, iC4, iC5, iC6, iC7, iCNN, iCPP, iCMAX};

#ifndef UNITTEST
#   define UNITTEST False
#endif

/***************************************** LOCAL POTENTIAL ************************************************/

static int cutoff;
static int lam;
static int channel;
static int T;

static int potential_print = 0;
static int order;
static double R0[5] = {0.8, 1.0, 1.2, 0.9, 1.1}; //different cutoffs
static double lambda[6] = {600.0/hbarc, 700.0/hbarc, 800.0/hbarc, 1000.0/hbarc, 1200.0/hbarc, 1400.0/hbarc}; //SFR cutoff
static double CS[5], CT[5], C1[5], C2[5], C3[5], C4[5], C5[5], C6[5], C7[5], CNN[5], CPP[5];
static double r[2];

static double ga=1.267;
static double gamod=1.29; // modified gA due to Goldberger-Treiman discrepancy, to be used for OPE at NLO and N2LO

// more constants
static double fpi=92.4/hbarc;
static double mpi=138.03/hbarc;
static double mpin=134.98/hbarc;
static double mpich=139.570/hbarc;
static double c1loc = -0.00081*hbarc;
static double c3loc = -0.00340*hbarc;
static double c4loc = 0.00340*hbarc;

static double deltaorder(int a) {if (a==0) {return(0.0);} else {return(1.0);}}
// delta function
static double deltafct(double r,int cutoff){return(exp(-pow(r/R0[cutoff],4)));}

static double long_reg(double r,int cutoff)
{

    //return pow((1.0-exp(-pow(r/R0[cutoff],2))),6); // MOD DRISCHLER JAN 2017
    return(1.0-exp(-pow(r/R0[cutoff],4))); // MOD DRISCHLER JAN 2017 without affecting anything
}

// smeared-out delta function for contact interactions
static double gfac(int cutoff){return(1.0/pow(R0[cutoff],3)/pi/gsl_sf_gamma(3.0/4.0));}
//normalization factor
static double yukawa(double r, double mpi){return(exp(-mpi*r)/mpi/r);}

// tensor operator in partial wave
static double s12(int S, int L, int L2, int J)
{
  if (S == 1)
  {
    if (L == J-1)
    {
      if (L2 == J-1) {return(-2.0*(J-1.0)/(2.0*J+1.0));}
      else if (L2 == J) {return(0.0);}
      else if (L2 == J+1) {return(6.0*sqrt(J*(J+1))/(2.0*J+1.0));}
      else {return(0.0);};
    }
    else if (L == J)
    {
      if (L2 == J-1) {return(0.0);}
      else if (L2 == J) {return(2.0);}
      else if (L2 == J+1) {return(0.0);}
      else {return(0.0);};
    }
    else if (L == J+1)
    {
      if (L2 == J-1) {return(6.0*sqrt(J*(J+1))/(2.0*J+1.0));}
      else if (L2 == J) {return(0.0);}
      else if (L2 == J+1) {return(-2.0*(J+2.0)/(2.0*J+1.0));}
      else {return(0.0);};
    }
    else {return(0.0);};
  }
  else {return(0.0);}
}

/******   Definition of spectral Functions for NLO TPE   ******/

static double etaC(double mu, double lambda)
{
    if (mu <= lambda) {return(1/768.0/pi/pow(fpi,4)*(4.0*mpi*mpi*(5.0*pow(ga,4)-4.0*pow(ga,2)-1)-mu*mu*(23.0*pow(ga,4)-10.0*pow(ga,2)-1)+48.0*pow(ga,4)*pow(mpi,4)/(4.0*mpi*mpi-mu*mu))*sqrt(mu*mu-4.0*mpi*mpi)/mu);}
    else {return(0.0);}
}

static double rhoS(double mu, double lambda)
{
    if (mu <= lambda) {return(3.0*pow(ga,4)/128.0/pi/pow(fpi,4)*mu*sqrt(mu*mu-4.0*mpi*mpi));}
    else return(0.0);
}

static double rhoT(double mu, double lambda)
{
    if (mu <= lambda) {return(3.0*pow(ga,4)/128.0/pi/pow(fpi,4)/mu*sqrt(mu*mu-4.0*mpi*mpi));}
    else {return(0.0);}
}


struct parametersSF {double r; double lambda;};
// NLO TPE spectral functions have to be integrated numerically over mu
// -> use GSL library
// integrated functions were checked against Alex Gezerlis code -> agreed very well

static double wcintegrand(double x, void *p)
{
    parametersSF *params= (parametersSF *)p;
    return (1.0/2.0/pi/pi*x*exp(-x*(params->r))*etaC(x,(params->lambda)));
}

static double vsintegrand(double x, void *p)
{
    parametersSF *params= (parametersSF *)p;
    return (1.0/3.0/pi/pi*x*exp(-x*(params->r))*rhoS(x,(params->lambda)));
}

static double vtintegrand(double x, void *p)
{
    parametersSF *params= (parametersSF *)p;
    return (-1.0/6.0/pi/pi*x*exp(-x*(params->r))*(3.0+3.0*x*(params->r)+x*x*(params->r)*(params->r))*rhoT(x,(params->lambda)));
}


static double wcfunc (double r, double lambda) //NLO TPE function Wc
{
    double error;
    double solution;
    size_t neval;

    gsl_function WC;
    parametersSF params = {r, lambda};

    WC.function = &wcintegrand;
    WC.params = &params;

    gsl_integration_qng(&WC, 2.0*mpi+0.0000001, lambda, 1, 1, &solution, &error, &neval);
    return (solution);
}

static double vsfunc (double r, double lambda) //NLO TPE function Vs
{
    double error;
    double solution;
    size_t neval;

    gsl_function VS;
    parametersSF params = {r, lambda};

    VS.function = &vsintegrand;
    VS.params = &params;

    gsl_integration_qng(&VS, 2.0*mpi+0.0000001, lambda, 1, 1e-4, &solution, &error, &neval);
    return (solution);
}

static double vtfunc (double r, double lambda)  //NLO TPE function Vt
{
    double error;
    double solution;
    size_t neval;

    gsl_function VT;
    parametersSF params = {r, lambda};

    VT.function = &vtintegrand;
    VT.params = &params;

    gsl_integration_qng(&VT, 2.0*mpi+0.0000001, lambda, 1, 1, &solution, &error, &neval);
    return (solution);
}


/******    Definition of potential contributions    *******/

/******    Definition of LO contributions    *******/

static double LO_IB_central (int cutoff, int channel)
{
    if (channel == -1) {return(CNN[cutoff]);}
    else if (channel == 0) {return(0.0);}
    else if (channel == 1) {return(CPP[cutoff]);}
    else return(0.0);
}

static void LO_IB_central (int cutoff, int channel, double *ret)
{
    if (channel == -1) {ret[iCNN] += (CNN[cutoff]);}
    else if (channel == 0) {ret[iVconst] += (0.0);}
    else if (channel == 1) {ret[iCPP] += (CPP[cutoff]);}
    else ret[iVconst] += (0.0);
}

static double LO_cont_central(double r, int cutoff, int channel)
{
    return((CS[cutoff]+LO_IB_central(cutoff,channel))*deltafct(r,cutoff)*gfac(cutoff));
}

static void LO_cont_central(double r, int cutoff, int channel, double *ret)
{
    double prefac = deltafct(r,cutoff)*gfac(cutoff);
    ret[iCS] += prefac * CS[cutoff];
    double tmp[16] = {};  // init to zero
    LO_IB_central(cutoff,channel,tmp);
    for(int i=iVconst; i<iCMAX; i++)
        ret[i] += prefac * tmp[i];
}

static double LO_cont_spin (double r, int cutoff)
{
    return(CT[cutoff]*deltafct(r,cutoff)*gfac(cutoff));
}

static void LO_cont_spin (double r, int cutoff, double *ret)
{
    ret[iCT] += (CT[cutoff]*deltafct(r,cutoff)*gfac(cutoff));
}


static double OPE_spin (double r, int cutoff, int channel, int T)
{
#if DOPION == 0
return 0;
#endif
    if ((channel == -1) || (channel == 1)) {return((+1/12.0/pi*gamod*gamod/4.0/fpi/fpi*pow(mpin,3)*yukawa(r,mpin))*long_reg(r,cutoff));}
    else if (channel == 0) {return((-1/12.0/pi*gamod*gamod/4.0/fpi/fpi*pow(mpin,3)*yukawa(r,mpin)+pow(-1.0,T+1)*2.0*1/12.0/pi*gamod*gamod/4.0/fpi/fpi*pow(mpich,3)*yukawa(r,mpich))*long_reg(r,cutoff));}
    else return(0.0);
}

static double OPE_tensor (double r, int cutoff, int channel, int T)
{
#if DOPION == 0
return 0;
#endif
    if ((channel == -1) || (channel == 1)) {return((+1/12.0/pi*gamod*gamod/4.0/fpi/fpi*pow(mpin,3)*(1.0+3.0/(mpin*r)+3.0/pow(mpin*r,2))*yukawa(r,mpin))*long_reg(r,cutoff));}
    else if (channel == 0) {return((-1/12.0/pi*gamod*gamod/4.0/fpi/fpi*pow(mpin,3)*(1.0+3.0/(mpin*r)+3.0/pow(mpin*r,2))*yukawa(r,mpin)+pow(-1.0,T+1)*2.0*1/12.0/pi*gamod*gamod/4.0/fpi/fpi*pow(mpich,3)*(1.0+3.0/(mpich*r)+3.0/pow(mpich*r,2))*yukawa(r,mpich))*long_reg(r,cutoff));}
    else return(0.0);
}



/******    Definition of NLO contributions    *******/

// NLO radial functions

static double lap_delta (double r, int cutoff)
{
    return(16.0*pow(r,6)/pow(R0[cutoff],8)-20.0*pow(r,2)/pow(R0[cutoff],4));
}

static double lap_delta_two (double r, int cutoff)
{
    return(8.0*pow(r,2)/pow(R0[cutoff],4)-16.0*pow(r,6)/pow(R0[cutoff],8));
}

// NLO contacts

static double NLO_cont_central (double r, int cutoff, int T)
{
    return(-(C1[cutoff]+(-3.0+2.0*T*(T+1))*C2[cutoff])*lap_delta(r,cutoff)*deltafct(r,cutoff)*gfac(cutoff));
}

static void NLO_cont_central (double r, int cutoff, int T, double *ret)
{
    double prefac = lap_delta(r,cutoff)*deltafct(r,cutoff)*gfac(cutoff);
    ret[iC1] -= C1[cutoff] * prefac;
    ret[iC2] -= (-3.0+2.0*T*(T+1))*C2[cutoff] * prefac;
}

static double NLO_cont_spin (double r, int cutoff, int T)
{
    return((-(C3[cutoff]+(-3.0+2.0*T*(T+1))*C4[cutoff])*lap_delta(r,cutoff)-1/3.0*(C6[cutoff]+(-3.0+2.0*T*(T+1))*C7[cutoff])*lap_delta(r,cutoff))*deltafct(r,cutoff)*gfac(cutoff));
}

static void NLO_cont_spin (double r, int cutoff, int T, double *ret)
{
    double prefac = deltafct(r,cutoff)*gfac(cutoff) * lap_delta(r,cutoff)  ;
    ret[iC3] -= C3[cutoff] * prefac;
    ret[iC4] -= (-3.0+2.0*T*(T+1))*C4[cutoff] * prefac;
    ret[iC6] -= 1/3.0*(C6[cutoff]) * prefac;
    ret[iC7] -= 1/3.0*((-3.0+2.0*T*(T+1))*C7[cutoff]) * prefac;
}

static double NLO_cont_ls (double r, int cutoff)
{
    return(2.0*C5[cutoff]*pow(r,2)/pow(R0[cutoff],4)*deltafct(r,cutoff)*gfac(cutoff));
}

static void NLO_cont_ls (double r, int cutoff, double *ret)
{
    ret[iC5] += (2.0*C5[cutoff]*pow(r,2)/pow(R0[cutoff],4)*deltafct(r,cutoff)*gfac(cutoff));
}

static double NLO_cont_tensor (double r, int cutoff, int T)
{
    return((C6[cutoff]+(-3.0+2.0*T*(T+1))*C7[cutoff])*1.0/3.0*lap_delta_two(r,cutoff)*deltafct(r,cutoff)*gfac(cutoff));
}

static void NLO_cont_tensor (double r, int cutoff, int T, double *ret)
{
    double prefac = 1.0/3.0*lap_delta_two(r,cutoff)*deltafct(r,cutoff)*gfac(cutoff);
    ret[iC6] += (C6[cutoff]) * prefac;
    ret[iC7] += ((-3.0+2.0*T*(T+1))*C7[cutoff]) * prefac;
}


// OPE at NLO, usually corrected, but our LO OPE already contains all corrections, see paper

static double OPE_CIB_spin (double r, int cutoff, int channel, int T)
{
#if DOPION == 0
return 0;
#endif
    return(OPE_spin(r, cutoff, channel, T));
}


static double OPE_CIB_tensor (double r, int cutoff, int channel, int T)
{
#if DOPION == 0
return 0;
#endif
    return(OPE_tensor(r, cutoff, channel, T));
}


// TPE at NLO

static double NLO_TPE_central (double r, int cutoff, int lam, int T)
{
#if DOTWOPION == 0
return 0;
#endif
    return(wcfunc(r,lambda[lam])/r*(-3.0+2.0*T*(T+1))*long_reg(r,cutoff));
}

static double NLO_TPE_spin (double r, int cutoff, int lam)
{
#if DOTWOPION == 0
return 0;
#endif
    return(vsfunc(r,lambda[lam])/r*long_reg(r,cutoff));
}

static double NLO_TPE_tensor (double r, int cutoff, int lam)
{
#if DOTWOPION == 0
return 0;
#endif
    return(vtfunc(r,lambda[lam])/pow(r,3)*long_reg(r,cutoff));
}


/******    Definition of N2LO contributions    *******/

static double NNLO_TPE_central (double r, int cutoff, int lam)
{
#if DOTWOPION == 0
return 0;
#endif
    return((3.0*ga*ga/32.0/pi/pi/pow(fpi,4)*exp(-2.0*mpi*r)/pow(r,6)*(2.0*c1loc*pow(mpi*r,2)*pow(1.0+mpi*r,2)
            +c3loc*(6.0+12.0*mpi*r+10.0*pow(mpi*r,2)+4.0*pow(mpi*r,3)+pow(mpi*r,4)))
            -3.0*ga*ga/128.0/pi/pi/pow(fpi,4)*exp(-lambda[lam]*r)/pow(r,6)*(4.0*c1loc*pow(mpi*r,2)*(2.0+lambda[lam]*r*(2.0+lambda[lam]*r)
            -2.0*pow(mpi*r,2))+c3loc*(24.0+lambda[lam]*r*(24.0+12.0*lambda[lam]*r
            +4.0*pow(lambda[lam]*r,2)+pow(lambda[lam]*r,3))-4.0*pow(mpi*r,2)*(2.0+2.0*lambda[lam]*r+pow(lambda[lam]*r,2))
            +4.0*pow(mpi*r,4))))*long_reg(r,cutoff));
}

static double NNLO_TPE_spin (double r, int cutoff, int lam, int T)
{
#if DOTWOPION == 0
return 0;
#endif
    return((-3.0+2.0*T*(T+1.0))*(ga*ga/48.0/pi/pi/pow(fpi,4)*exp(-2*mpi*r)/pow(r,6)*c4loc*(1.0+mpi*r)*(3.0+3.0*mpi*r+2.0*pow(mpi*r,2))
            -ga*ga/384.0/pi/pi/pow(fpi,4)*exp(-lambda[lam]*r)/pow(r,6)*c4loc*(24.0+24.0*lambda[lam]*r
            +12.0*pow(lambda[lam]*r,2)+4.0*pow(lambda[lam]*r,3)+pow(lambda[lam]*r,4)
            -4.0*pow(mpi*r,2)*(2.0+2.0*lambda[lam]*r+pow(lambda[lam]*r,2))))*long_reg(r,cutoff));
}

static double NNLO_TPE_tensor (double r, int cutoff, int lam, int T)
{
#if DOTWOPION == 0
return 0;
#endif
    return((-3.0+2.0*T*(T+1.0))*(-ga*ga/48.0/pi/pi/pow(fpi,4)*exp(-2.0*mpi*r)/pow(r,6)*c4loc*(1.0+mpi*r)*(3.0+3.0*mpi*r
            +pow(mpi*r,2))+ga*ga/768.0/pi/pi/pow(fpi,4)*exp(-lambda[lam]*r)/pow(r,6)*c4loc*(48.0+48.0*lambda[lam]*r
            +24.0*pow(lambda[lam]*r,2)+7.0*pow(lambda[lam]*r,3)+pow(lambda[lam]*r,4)
            -4.0*pow(mpi*r,2)*(8.0+5.0*lambda[lam]*r+pow(lambda[lam]*r,2))))*long_reg(r,cutoff));
}


/******    Definition of total potential    *******/

// LO
static double LOcentral (double r, int cutoff, int channel)
{
    return(LO_cont_central(r, cutoff, channel));
}

static void LOcentral (double r, int cutoff, int channel, double *ret)
{
    LO_cont_central(r, cutoff, channel, ret);
}

static double LOspin (double r, int cutoff, int order, int channel, int T)
{
    return(LO_cont_spin(r,cutoff) + (1-deltaorder(order))*OPE_spin(r,cutoff,channel,T));
}

static void LOspin (double r, int cutoff, int order, int channel, int T, double *ret)
{
    double OBE_spin_contr = (1-deltaorder(order))*OPE_spin(r,cutoff,channel,T);
    LO_cont_spin(r,cutoff, ret);
    ret[iVconst] += OBE_spin_contr;
}

static double LOtensor (double r, int cutoff, int order, int channel, int T)
{
    return((1-deltaorder(order))*OPE_tensor(r,cutoff,channel,T));
}

static void LOtensor (double r, int cutoff, int order, int channel, int T, double *ret)
{
    double tmp = ((1-deltaorder(order))*OPE_tensor(r,cutoff,channel,T));
    ret[iVconst] += tmp;
}

// NLO
static double NLOcentral (double r, int cutoff, int lam, int T)
{
    return(NLO_cont_central(r, cutoff,T) + NLO_TPE_central(r,cutoff, lam,T));
}

static void NLOcentral (double r, int cutoff, int lam, int T, double *ret)
{
    NLO_cont_central(r, cutoff,T, ret);
    double NLO_TPE_central_contr = NLO_TPE_central(r,cutoff, lam,T);
    ret[iVconst] += NLO_TPE_central_contr;
}

static double NLOspin (double r, int cutoff, int lam, int channel,int T)
{
    return(NLO_cont_spin(r,cutoff,T) + OPE_CIB_spin(r,cutoff,channel,T) + NLO_TPE_spin(r,cutoff, lam));
}

static void NLOspin (double r, int cutoff, int lam, int channel,int T, double *ret)
{
    NLO_cont_spin(r,cutoff,T, ret) ;
    double pion_ex_contr = OPE_CIB_spin(r,cutoff,channel,T) + NLO_TPE_spin(r,cutoff, lam);
    ret[iVconst] += pion_ex_contr;
}

static double NLOls (double r, int cutoff)
{
    return(NLO_cont_ls(r,cutoff));
}

static void NLOls (double r, int cutoff, double *ret)
{
    NLO_cont_ls(r,cutoff, ret);
}

static double NLOtensor (double r, int cutoff, int lam, int channel, int T)
{
    return(NLO_cont_tensor(r,cutoff,T) + OPE_CIB_tensor(r,cutoff,channel,T) + NLO_TPE_tensor(r,cutoff, lam));
}

static void NLOtensor (double r, int cutoff, int lam, int channel, int T, double *ret)
{
    NLO_cont_tensor(r,cutoff,T, ret) ;
    double pion_ex_contr = OPE_CIB_tensor(r,cutoff,channel,T) + NLO_TPE_tensor(r,cutoff, lam);
   ret[iVconst] += pion_ex_contr;
}

// add everything up
// L=L2
static double Vfull (double r, int cutoff, int lam, int order, int S, int L, int J, int T, int channel)
{
    return(//LO potential
           LOcentral(r,cutoff,channel) + (-3.0 + 2.0*S*(S+1.0)) * LOspin(r,cutoff,order,channel,T) + s12(S,L,L,J) * LOtensor(r,cutoff,order,channel,T)
           // NLO potential
            + deltaorder(order)*(NLOcentral(r,cutoff,lam,T) + (-3.0 + 2.0*S*(S+1.0)) * NLOspin(r,cutoff,lam,channel,T)
            + (J*(J+1.0)-L*(L+1.0)-S*(S+1.0))/2.0 * NLOls(r,cutoff) + s12(S,L,L,J) * NLOtensor(r,cutoff,lam, channel,T))
           // N2LO potential
            + deltaorder(order)*(order-1)*(NNLO_TPE_central (r, cutoff, lam) + (-3.0+2.0*S*(S+1.0)) * NNLO_TPE_spin (r, cutoff, lam, T)
            + s12(S,L,L,J) * NNLO_TPE_tensor (r, cutoff, lam, T)));
}

// L=L2
static void Vfull(double r, int cutoff, int lam, int order, int S, int L, int J, int T, int channel, double *ret)
{
    double arr[256] = {};
    double *LOcentral_arr = arr;
    double *LOspin_arr = &arr[16];
    double *LOtensor_arr = &arr[2*16];

    double *NLOcentral_arr = &arr[3*16];
    double *NLOspin_arr = &arr[4*16];
    double *NLOls_arr = &arr[5*16];
    double *NLOtensor_arr = &arr[6*16];

    LOcentral(r,cutoff,channel, LOcentral_arr);
    LOspin(r,cutoff,order,channel,T, LOspin_arr);
    LOtensor(r,cutoff,order,channel,T, LOtensor_arr);

    NLOcentral(r,cutoff,lam,T, NLOcentral_arr );
    NLOspin(r,cutoff,lam,channel,T, NLOspin_arr);
    NLOls(r,cutoff, NLOls_arr) ;
    NLOtensor(r,cutoff,lam, channel,T, NLOtensor_arr);

    double prefac_nlo = deltaorder(order);
    double prefac_n2lo = deltaorder(order)*(order-1);
    double internal_ret[16] = {};

    // lots could be optimized here, but we want to keep the structure of the original code
    for(int i = iVconst; i< iCMAX; i++){
        internal_ret[i] += LOcentral_arr[i];
        internal_ret[i] += (-3.0 + 2.0*S*(S+1.0)) * LOspin_arr[i];
        internal_ret[i] += s12(S,L,L,J) * LOtensor_arr[i];

        internal_ret[i] += prefac_nlo * NLOcentral_arr[i];
        internal_ret[i] += prefac_nlo * (-3.0 + 2.0*S*(S+1.0)) * NLOspin_arr[i];
        internal_ret[i] += prefac_nlo * (J*(J+1.0)-L*(L+1.0)-S*(S+1.0))/2.0 *  NLOls_arr[i];
        internal_ret[i] += prefac_nlo * s12(S,L,L,J) * NLOtensor_arr[i];
    }

    internal_ret[iVconst] += prefac_n2lo * NNLO_TPE_central(r, cutoff, lam);
    internal_ret[iVconst] += prefac_n2lo *(-3.0+2.0*S*(S+1.0)) *  NNLO_TPE_spin(r, cutoff, lam, T);
    internal_ret[iVconst] += prefac_n2lo * s12(S,L,L,J) * NNLO_TPE_tensor(r, cutoff, lam, T);

    double sum = 0.;
    for(int i = iVconst; i< iCMAX; i++) {
        sum += internal_ret[i];
        ret[i] += internal_ret[i];
    }

#if UNITTEST
    double sum_ref = Vfull(r, cutoff, lam, order, S, L, J, T, channel);
    double threshold = 1e-14;
    // printf("test Vfull: %lf, %lf | %.4e\n", sum, sum_ref, sum - sum_ref);
    if(fabs(sum - sum_ref) > threshold){
        printf("error: mismatch in (affine) `Vfull`\n");
        exit(EXIT_FAILURE);
    }
#endif
}

//L neq L2
static double Vfull_tensor (double r, int cutoff, int lam, int order, int S, int L, int LL, int J, int T, int channel)
{
    return(s12(S,L,LL,J) * (LOtensor (r,cutoff,order,channel,T)
           + deltaorder(order) * NLOtensor(r,cutoff,lam,channel, T)
           + deltaorder(order) * (order-1) * NNLO_TPE_tensor (r, cutoff, lam,T)));
}

//L neq L2
static void Vfull_tensor(double r, int cutoff, int lam, int order, int S, int L, int LL, int J, int T, int channel, double *ret)
{
    double arr[64] = {};
    double *LOtensor_arr = arr;
    double *NLOtensor_arr = arr+16;
    LOtensor (r,cutoff,order,channel,T, LOtensor_arr);
    NLOtensor(r,cutoff,lam,channel, T, NLOtensor_arr);

    double prefac = s12(S,L,LL,J);
    double internal_ret[16] = {};
    for(int i = iVconst; i< iCMAX; i++){
        internal_ret[i] += LOtensor_arr[i] * prefac;
        internal_ret[i] += NLOtensor_arr[i] * prefac * deltaorder(order);

    }
    internal_ret[iVconst] += NNLO_TPE_tensor(r, cutoff, lam,T) * prefac * deltaorder(order) * (order-1);

    double sum = 0;
    for(int i = iVconst; i< iCMAX; i++){
         sum += internal_ret[i];
         ret[i] += internal_ret[i];
    }

#if UNITTEST
    double sum_ref = Vfull_tensor(r, cutoff, lam, order, S, L, LL, J, T, channel);
    double threshold = 1e-15;
    // printf("test Vfull_tensor: %lf, %lf\n", sum, sum_ref);
    if(fabs(sum - sum_ref) > threshold){
        printf("error: mismatch in (affine) `Vfull_tensor`\n");
        exit(EXIT_FAILURE);
    }
#endif
}



/*********************************** Local potential ****************************************/

struct parameters {double k; double kk; int cutoff; int S; int L; int LL; int J; int T; int order; int lam; int channel;};

// integrand for Fourier trafo, with all LEC definitions
static double NNLOintegrand(double r, void *p)
{
    parameters *params= (parameters *)p;

    if (potential_print)
    {
        potential_print--;
        if (params->order == 0) {cout << "Local chiral potential at LO: cutoff " << R0[params->cutoff] << " fm, lambda " << lambda[params->lam] << " fm." << endl;}
        else if (params->order == 1) {cout << "Local chiral potential at NLO: cutoff " << R0[params->cutoff] << " fm, lambda " << lambda[params->lam] << " fm." << endl;}
        else if (params->order == 2) {cout << "Local chiral potential at NNLO: cutoff " << R0[params->cutoff] << " fm, lambda " << lambda[params->lam] << " fm." << endl;}
        else { cout << "not" << endl;}
    }



    if ((params->lam) == 2) { //SFR 800 MeV
        if ((params->order)==0) { // LO
            CS[0]=0.63603; CS[1]=-0.75112; CS[2]=-1.79693; CS[3]=-0.12898; CS[4]=-1.29631;
            CT[0]=0.71047; CT[1]=0.37409; CT[2]=0.15442; CT[3]=0.51809; CT[4]=0.25648;
            CNN[0]=-0.00996; CNN[1]=-0.00373; CNN[2]=0.00624; CNN[3]=-0.00732; CNN[4]=0.00079;
            CPP[0]=0.0; CPP[1]=-0.04349; CPP[2]=-0.03294; CPP[3]=0.0; CPP[4]=-0.03923;    }
        else if ((params->order)==1){ // NLO, no numbers for 0.8 fm
            CS[0]=0.0; CS[1]=3.06894; CS[2]=-0.02682; CS[3]=9.37208; CS[4]=0.94811;
            CT[0]=0.0; CT[1]=1.50714; CT[2]=0.77728; CT[3]=3.24120; CT[4]=0.98044;
            C1[0]=0.0; C1[1]=0.32552; C1[2]=0.23418; C1[3]=0.37370; C1[4]=0.28633;
            C2[0]=0.0; C2[1]=0.25322; C2[2]=0.22115; C2[3]=0.36743; C2[4]=0.21102;
            C3[0]=0.0; C3[1]=-0.13085; C3[2]=-0.14504; C3[3]=-0.18147; C3[4]=-0.13546;
            C4[0]=0.0; C4[1]=0.11057; C4[2]=0.07997; C4[3]=-0.20436; C4[4]=0.08746;
            C5[0]=0.0; C5[1]=-2.38561; C5[2]=-2.03030; C5[3]=-2.99774; C5[4]=-2.16193;
            C6[0]=0.0; C6[1]=0.36812; C6[2]=0.33264; C6[3]=0.54104; C6[4]=0.32363;
            C7[0]=0.0; C7[1]=-0.33952; C7[2]=-0.34508; C7[3]=-0.48576; C7[4]=-0.31762;
            CNN[0]=0.0; CNN[1]=0.04271; CNN[2]=0.04818; CNN[3]=0.04680; CNN[4]=0.04449;
            CPP[0]=0.0; CPP[1]=0.05917; CPP[2]=0.06137; CPP[3]=0.0; CPP[4]=0.05857;    }
        else if ((params->order)==2) { // N2LO
            CS[0]=9.67288; CS[1]=3.82785; CS[2]=1.66533; CS[3]=5.87393; CS[4]=2.60023;
            CT[0]=1.68398; CT[1]=0.55793; CT[2]=0.40535; CT[3]=0.86404; CT[4]=0.46074;
            C1[0]=-0.20020; C1[1]=-0.08163; C1[2]=-0.07173; C1[3]=-0.11819; C1[4]=-0.06749;
            C2[0]=0.08402; C2[1]=0.06878; C2[2]=0.09887; C2[3]=0.06974; C2[4]=0.08512;
            C3[0]=-0.13785; C3[1]=-0.11667; C3[2]=-0.15648; C3[3]=-0.11547; C3[4]=-0.12997;
            C4[0]=0.12427; C4[1]=0.08344; C4[2]=0.09975; C4[3]=0.09570; C4[4]=0.08398;
            C5[0]=-3.29260; C5[1]=-2.11922; C5[2]=-1.94474; C5[3]=-2.45457; C5[4]=-2.01259;
            C6[0]=0.23903; C6[1]=0.18114; C6[2]=0.23089; C6[3]=0.18957; C6[4]=0.20200;
            C7[0]=-0.30159; C7[1]=-0.25143; C7[2]=-0.32675; C7[3]=-0.25349; C7[4]=-0.28380;
            CNN[0]=0.04249; CNN[1]=0.04344; CNN[2]=0.04879; CNN[3]=0.04209; CNN[4]=0.04636;
            CPP[0]=0.0; CPP[1]=0.06296; CPP[2]=0.06419; CPP[3]=0.0; CPP[4]=0.0644;    }
        else {cout << "Error with LECs!" << endl;}
    }
    else if ((params->lam) == 3) { //SFR 1000 MeV



        if ((params->order)==0) { // LO
            CS[0]=0.63603; CS[1]=-0.75112; CS[2]=-1.79693; CS[3]=-0.12898; CS[4]=-1.29631;
            CT[0]=0.71047; CT[1]=0.37409; CT[2]=0.15442; CT[3]=0.51809; CT[4]=0.25648;
            CNN[0]=-0.00996; CNN[1]=-0.00373; CNN[2]=0.00624; CNN[3]=-0.00732; CNN[4]=0.00079;
            CPP[0]=0.0; CPP[1]=-0.04349; CPP[2]=-0.03294; CPP[3]=-0.045768; CPP[4]=-0.03923;    }

        else if ((params->order)==1){ // NLO
            CS[0]=0.0; CS[1]=3.16803; CS[2]=0.03551; CS[3]=9.55049; CS[4]=1.03075;
            CT[0]=0.0; CT[1]=1.41396; CT[2]=0.71729; CT[3]=3.14878; CT[4]=0.90699;
            C1[0]=0.0; C1[1]=0.31420; C1[2]=0.22288; C1[3]=0.37085; C1[4]=0.27239;
            C2[0]=0.0; C2[1]=0.25786; C2[2]=0.22878; C2[3]=0.36361; C2[4]=0.22032;
            C3[0]=0.0; C3[1]=-0.13134; C3[2]=-0.15043; C3[3]=-0.18432; C3[4]=-0.13641;
            C4[0]=0.0; C4[1]=0.11861; C4[2]=0.08929; C4[3]=0.21821; C4[4]=0.09420;
            C5[0]=0.0; C5[1]=-2.38552; C5[2]=-2.02932; C5[3]=-3.00330; C5[4]=-2.16238;
            C6[0]=0.0; C6[1]=0.37319; C6[2]=0.34011; C6[3]=0.54373; C6[4]=0.33065;
            C7[0]=0.0; C7[1]=-0.35668; C7[2]=-0.36248; C7[3]=-0.50259; C7[4]=-0.33570;
            CNN[0]=0.0; CNN[1]=0.04271; CNN[2]=0.04817; CNN[3]=0.04655; CNN[4]=0.04449;
            CPP[0]=0.0; CPP[1]=0.059175; CPP[2]=0.061364; CPP[3]=0.069507; CPP[4]=0.05857;    }

        else if ((params->order)==2) { // N2LO
            CS[0]=11.0104; CS[1]=5.43850; CS[2]=2.68765; CS[3]=7.74784; CS[4]=3.88699;
            CT[0]=1.16562; CT[1]=0.27672; CT[2]=0.23382; CT[3]=0.45247; CT[4]=0.24416;
            C1[0]=-0.35339; C1[1]=-0.14084; C1[2]=-0.07951; C1[3]=-0.21715; C1[4]=-0.09650;
            C2[0]=0.04565; C2[1]=0.04243; C2[2]=0.07610; C2[3]=0.03457; C2[4]=0.05947;
            C3[0]=-0.13323; C3[1]=-0.12338; C3[2]=-0.16926; C3[3]=-0.11535; C3[4]=-0.14183;
            C4[0]=0.15118; C4[1]=0.11018; C4[2]=0.12359; C4[3]=0.11818; C4[4]=0.11146;
            C5[0]=-3.30610; C5[1]=-2.11254; C5[2]=-1.94280; C5[3]=-2.41603; C5[4]=-2.0082;
            C6[0]=0.19459; C6[1]=0.15898; C6[2]=0.21421; C6[3]=0.15463; C6[4]=0.18318;
            C7[0]=-0.32546; C7[1]=-0.26994; C7[2]=-0.34193; C7[3]=-0.26709; C7[4]=-0.30105;
            CNN[0]=0.04010; CNN[1]=0.04344; CNN[2]=0.04877; CNN[3]=0.04164; CNN[4]=0.04636;
            CPP[0]=0.0; CPP[1]=0.062963; CPP[2]=0.064192; CPP[3]=0.06329; CPP[4]=0.064395; }
        else {cout << "Error with LECs!" << endl;}




    }
    else if ((params->lam) == 4) { // SFR 1200 MeV
        if ((params->order)==0) { // LO
            CS[0]=0.63603; CS[1]=-0.75112; CS[2]=-1.79693; CS[3]=-0.12898; CS[4]=-1.29631;
            CT[0]=0.71047; CT[1]=0.37409; CT[2]=0.15442; CT[3]=0.51809; CT[4]=0.25648;
            CNN[0]=-0.00996; CNN[1]=-0.00373; CNN[2]=0.00624; CNN[3]=-0.00732; CNN[4]=0.00079;
            CPP[0]=0.0; CPP[1]=-0.04349; CPP[2]=-0.03294; CPP[3]=0.0; CPP[4]=-0.03923;    }
        else if ((params->order)==1){ // NLO
            CS[0]=0.0; CS[1]=3.31668; CS[2]=0.07785; CS[3]=9.82149; CS[4]=1.09179;
            CT[0]=0.0; CT[1]=1.36887; CT[2]=0.67523; CT[3]=3.11612; CT[4]=0.85469;
            C1[0]=0.0; C1[1]=0.31002; C1[2]=0.21645; C1[3]=0.36985; C1[4]=0.26360;
            C2[0]=0.0; C2[1]=0.26539; C2[2]=0.23507; C2[3]=0.36772; C2[4]=0.22868;
            C3[0]=0.0; C3[1]=-0.13791; C3[2]=-0.15778; C3[3]=-0.19130; C3[4]=-0.14046;
            C4[0]=0.0; C4[1]=0.12949; C4[2]=0.09665; C4[3]=0.23153; C4[4]=0.09962;
            C5[0]=0.0; C5[1]=-2.39935; C5[2]=-2.02844; C5[3]=-3.02092; C5[4]=-2.16343;
            C6[0]=0.0; C6[1]=0.38404; C6[2]=0.34681; C6[3]=0.55478; C6[4]=0.33779;
            C7[0]=0.0; C7[1]=-0.37351; C7[2]=-0.37402; C7[3]=-0.52057; C7[4]=-0.34834;
            CNN[0]=0.0; CNN[1]=0.04270; CNN[2]=0.04817; CNN[3]=0.04649; CNN[4]=0.04449;
            CPP[0]=0.0; CPP[1]=0.0; CPP[2]=0.0; CPP[3]=0.0; CPP[4]=0.0;    }
        else if ((params->order)==2) { // N2LO
            CS[0]=0.0; CS[1]=6.91212; CS[2]=3.52998; CS[3]=9.74348; CS[4]=4.99755;
            CT[0]=0.0; CT[1]=0.04285; CT[2]=0.09195; CT[3]=0.14859; CT[4]=0.06480;
            C1[0]=0.0; C1[1]=-0.14827; C1[2]=-0.04889; C1[3]=-0.25208; C1[4]=-0.08146;
            C2[0]=0.0; C2[1]=0.02485; C2[2]=0.05752; C2[3]=0.01733; C2[4]=0.04084;
            C3[0]=0.0; C3[1]=-0.12777; C3[2]=-0.17849; C3[3]=-0.11688; C3[4]=-0.14957;
            C4[0]=0.0; C4[1]=0.12246; C4[2]=0.13115; C4[3]=0.13337; C4[4]=0.12215;
            C5[0]=0.0; C5[1]=-2.11771; C5[2]=-1.94801; C5[3]=-2.43112; C5[4]=-2.0140;
            C6[0]=0.0; C6[1]=0.14181; C6[2]=0.19905; C6[3]=0.13688; C6[4]=0.16701;
            C7[0]=0.0; C7[1]=-0.27729; C7[2]=-0.34594; C7[3]=-0.27811; C7[4]=-0.30685;
            CNN[0]=0.0; CNN[1]=0.04339; CNN[2]=0.04875; CNN[3]=0.04143; CNN[4]=0.04641;
            CPP[0]=0.0; CPP[1]=0.0; CPP[2]=0.0; CPP[3]=0.0; CPP[4]=0.0;    }
        else {cout << "Error with LECs!" << endl;}
    }
    else if ((params->lam) == 5) { // SFR 1400 MeV
        if ((params->order)==0) { // LO
            CS[0]=0.63603; CS[1]=-0.75112; CS[2]=-1.79693; CS[3]=-0.12898; CS[4]=-1.29631;
            CT[0]=0.71047; CT[1]=0.37409; CT[2]=0.15442; CT[3]=0.51809; CT[4]=0.25648;
            CNN[0]=-0.00996; CNN[1]=-0.00373; CNN[2]=0.00624; CNN[3]=-0.00732; CNN[4]=0.00079;
            CPP[0]=0.0; CPP[1]=-0.04349; CPP[2]=-0.03294; CPP[3]=0.0; CPP[4]=-0.03923;    }
        else if ((params->order)==1){ // NLO
            CS[0]=0.0; CS[1]=3.32404; CS[2]=0.10909; CS[3]=9.84285; CS[4]=1.13903;
            CT[0]=0.0; CT[1]=1.30221; CT[2]=0.64646; CT[3]=3.03064; CT[4]=0.81867;
            C1[0]=0.0; C1[1]=0.30649; C1[2]=0.21280; C1[3]=0.36791; C1[4]=0.25830;
            C2[0]=0.0; C2[1]=0.26558; C2[2]=0.24032; C2[3]=0.36680; C2[4]=0.23565;
            C3[0]=0.0; C3[1]=-0.14378; C3[2]=-0.16477; C3[3]=-0.19579; C3[4]=-0.14535;
            C4[0]=0.0; C4[1]=0.13434; C4[2]=0.10228; C4[3]=0.23658; C4[4]=0.10401;
            C5[0]=0.0; C5[1]=-2.39094; C5[2]=-2.02827; C5[3]=-3.01594; C5[4]=-2.16525;
            C6[0]=0.0; C6[1]=0.38680; C6[2]=0.35219; C6[3]=0.55714; C6[4]=0.34394;
            C7[0]=0.0; C7[1]=-0.37920; C7[2]=-0.38191; C7[3]=-0.52609; C7[4]=-0.35731;
            CNN[0]=0.0; CNN[1]=0.04267; CNN[2]=0.04817; CNN[3]=0.04642; CNN[4]=0.04450;
            CPP[0]=0.0; CPP[1]=0.05909; CPP[2]=0.06120; CPP[3]=0.0; CPP[4]=0.05852;    }
        else if ((params->order)==2) { // N2LO
            CS[0]=0.0; CS[1]=8.16454; CS[2]=4.19629; CS[3]=11.4902; CS[4]=5.89685;
            CT[0]=0.0; CT[1]=-0.14809; CT[2]=-0.02820; CT[3]=-0.10848; CT[4]=-0.08689;
            C1[0]=0.0; C1[1]=-0.12250; C1[2]=0.00211; C1[3]=-0.24666; C1[4]=-0.04061;
            C2[0]=0.0; C2[1]=0.00843; C2[2]=0.03805; C2[3]=0.00136; C2[4]=0.02161;
            C3[0]=0.0; C3[1]=-0.12964; C3[2]=-0.18525; C3[3]=-0.11635; C3[4]=-0.15446;
            C4[0]=0.0; C4[1]=0.12390; C4[2]=0.12819; C4[3]=0.13700; C4[4]=0.12110;
            C5[0]=0.0; C5[1]=-2.13434; C5[2]=-1.95804;  C5[3]=-2.44931; C5[4]=-2.02482;
            C6[0]=0.0; C6[1]=0.12495; C6[2]=0.18335; C6[3]=0.11906; C6[4]=0.14992;
            C7[0]=0.0; C7[1]=-0.27533; C7[2]=-0.34227; C7[3]=-0.27722; C7[4]=-0.30346;
            CNN[0]=0.0; CNN[1]=0.04328; CNN[2]=0.04878; CNN[3]=0.04118; CNN[4]=0.04655;
            CPP[0]=0.0; CPP[1]=0.06251; CPP[2]=0.06306; CPP[3]=0.0; CPP[4]=0.06421; }
        else {cout << "Error with LECs!" << endl;}
    }

#if DOCONT == 0
    for( int i=0; i<5; i++ )
    {
        CS[ i ] = 0. ;
        CT[ i ] = 0. ;
        CNN[ i ] = 0. ;
        CPP[ i ]  = 0. ;
        C1[ i ] = 0. ;
        C2[ i ] = 0. ;
        C3[ i ] = 0. ;
        C4[ i ] = 0. ;
        C5[ i ] = 0. ;
        C6[ i ] = 0. ;
        C7[ i ] = 0. ;
    }
#endif

    if( output_Vr )
    {
        if ((params->L)==(params->LL))
        {
            return ( Vfull(r, params->cutoff, params->lam, params->order, params->S, params->L, params->J,params->T,params->channel) );
        }
        else
        {
            return ( Vfull_tensor (r, params->cutoff, params->lam, params->order, params->S, params->L, params->LL, params->J,params->T,params->channel) );
        }
    }

    if ((params->L)==(params->LL))
    {
        return (r*r*gsl_sf_bessel_jl((params->L),(params->k)*r) * gsl_sf_bessel_jl((params->L),(params->kk)*r) * Vfull(r, params->cutoff, params->lam, params->order, params->S, params->L, params->J,params->T,params->channel));
    }
    else
    {
        return (r*r*gsl_sf_bessel_jl((params->L),(params->k)*r) * gsl_sf_bessel_jl((params->LL),(params->kk)*r) * Vfull_tensor (r, params->cutoff, params->lam, params->order, params->S, params->L, params->LL, params->J,params->T,params->channel));
    }
}


// return V(r) for a given channel and set of NN LECs
double Vrlocal(double r, int pot, Channel *chan, Lecs *lecs)
{
  int S=chan->S; int L=chan->L; int LL=chan->LL;
  int J=chan->J; int channel=chan->channel;

  // determine chiral order and cutoffs
  // pot: xyz:  x-> order: LO=0, NLO=1, N2LO=2
  //            y-> cutoff: 0.8=0, 1.0=1, 1.2=2, 0.9=3, 1.1=4
  //            z-> SFR cutoff: 800=2, 1000=3, 1200=4, 1400=5
  // Channel: nn -> -1, np -> 0, pp -> 1

  int order_in, cutoff_in, lambda_in;
  double ret;

  order_in = pot/100;
  cutoff_in = (pot-100*(pot/100))/10;
  lambda_in = pot-10*(pot/10);

  if (potential_print)
  {
      potential_print--;
      if (order_in == 0) {cout << "Local chiral potential at LO: cutoff " << R0[cutoff_in] << " fm, lambda " << lambda[lambda_in] << " fm." << endl;}
      else if (order_in == 1) {cout << "Local chiral potential at NLO: cutoff " << R0[cutoff_in] << " fm, lambda " << lambda[lambda_in] << " fm." << endl;}
      else if (order_in == 2) {cout << "Local chiral potential at NNLO: cutoff " << R0[cutoff_in] << " fm, lambda " << lambda[lambda_in] << " fm." << endl;}
      else { cout << "not" << endl;}
  }

  // determine isospin channel T
  if (L % 2 == 0) {
      if (S == 0) {T = 1;}
      else if (S == 1) {T = 0;}
  }
  else if (L % 2 == 1) {
      if (S == 0) {T = 0;}
      else if (S == 1) {T = 1;}
  }

  // fill (global) arrays with input LECs
  // note that setting only the component [cutoff_in] would suffice
  for(int i=0; i<5; ++i){
    CS[i] = lecs->CS; // fm^2
    CT[i] = lecs->CT; // fm^2
    C1[i] = lecs->C1; // fm^4
    C2[i] = lecs->C2;
    C3[i] = lecs->C3;
    C4[i] = lecs->C4;
    C5[i] = lecs->C5;
    C6[i] = lecs->C6;
    C7[i] = lecs->C7;
    CNN[i] = lecs->CNN;
    CPP[i] = lecs->CPP;
  }

    // simple debugging
    //printf("LECs: CS %lf; CT %lf | C1 %lf; C2 %lf; C3 %lf; C4 %lf; C5 %lf; C6 %lf; C7 %lf; CNN %lf; CPP %lf\n",
    //lecs->CS, lecs->CT, lecs->C1, lecs->C2, lecs->C3,
    //lecs->C4, lecs->C5, lecs->C6, lecs->C7, lecs->CNN, lecs->CPP);

    // printf("S: %d; L: %d; LL: %d; J: %d;\n", S, L, LL, J);

  // check that potential exists and evaluate
  if (pot == 0) {
      strcpy(interaction, "No NN potential");
      ret=0.0;
  }
  else if ((order_in < 3) && (cutoff_in < 5) && (lambda_in < 6)) { // Local potential
    strcpy(interaction,"Local potential");
    double prefactor = hbarc;  // we want the potential in MeV (not fm^(-1))

    if ((L)==(LL))
    {
        return ( prefactor*Vfull(r, cutoff_in, lambda_in, order_in, S, L, J,T,channel) );
    }
    else
    {
        return ( prefactor*Vfull_tensor (r, cutoff_in, lambda_in, order_in, S, L, LL, J,T,channel) );
    }
  }
  else {
      unknown_potential = 1;
      ret=0.0;
  }
  return ret;

}

// return V(r) for a given channel and set of NN LECs
void Vrlocal_affine(double r, int pot, Channel *chan, double *ret)
{
  int S=chan->S; int L=chan->L; int LL=chan->LL;
  int J=chan->J; int channel=chan->channel;

  // determine chiral order and cutoffs
  // pot: xyz:  x-> order: LO=0, NLO=1, N2LO=2
  //            y-> cutoff: 0.8=0, 1.0=1, 1.2=2, 0.9=3, 1.1=4
  //            z-> SFR cutoff: 800=2, 1000=3, 1200=4, 1400=5
  // Channel: nn -> -1, np -> 0, pp -> 1

  int order_in, cutoff_in, lambda_in;
  double internal_output[16] = {};

  order_in = pot/100;
  cutoff_in = (pot-100*(pot/100))/10;
  lambda_in = pot-10*(pot/10);

  // determine isospin channel T
  if (L % 2 == 0) {
      if (S == 0) {T = 1;}
      else if (S == 1) {T = 0;}
  }
  else if (L % 2 == 1) {
      if (S == 0) {T = 0;}
      else if (S == 1) {T = 1;}
  }

  // fill (global) arrays with input LECs
  // note that setting only the component [cutoff_in] would suffice
  double fence_value = 1.;  // has to be one because we keep the multiplication by the LECs intact
  for(int i=0; i<5; ++i){
    CS[i] = fence_value; // fm^2
    CT[i] = fence_value; // fm^2
    C1[i] = fence_value; // fm^4
    C2[i] = fence_value;
    C3[i] = fence_value;
    C4[i] = fence_value;
    C5[i] = fence_value;
    C6[i] = fence_value;
    C7[i] = fence_value;
    CNN[i] = fence_value;
    CPP[i] = fence_value;
  }

  // check that potential exists and evaluate
  if (pot == 0) {
      for(int i = iVconst; i<iCMAX; i++) internal_output[i] = 0.0;  // not needed but let's keep the old code's structure
  }
  else if ((order_in < 3) && (cutoff_in < 5) && (lambda_in < 6)) { // Local potential
    if ((L)==(LL)){
        Vfull(r, cutoff_in, lambda_in, order_in, S, L, J,T,channel, internal_output);
    }
    else {
        Vfull_tensor(r, cutoff_in, lambda_in, order_in, S, L, LL, J,T,channel, internal_output);
    }
  }
  else {
      unknown_potential = 1;
      for(int i = iVconst; i<iCMAX; i++) internal_output[i] = 0.0;  // not needed but let's keep the old code's structure
  }
    double prefactor = hbarc;  // we want the potential in MeV (not fm^(-1))    
    for(int i = iVconst; i<iCMAX; i++) ret[i] = prefactor * internal_output[i];

}



static double Vlocal (double k, double kk, int S, int L, int LL, int J, int order, int cutoffpar, int lambdapar, int channelpar)
{
    double sol;
    double error;

    cutoff = cutoffpar;
    lam = lambdapar;
    channel = channelpar;

    if (L % 2 == 0) {
        if (S == 0) {T = 1;}
        else if (S == 1) {T = 0;}
    }
    else if (L % 2 == 1) {
        if (S == 0) {T = 0;}
        else if (S == 1) {T = 1;}
    }


#if DOCONT == 0 || DOTWOPION == 0 || DOPION == 0
//if (potential_print) cout << "NOTICE: no contacts!\n" << endl;
#endif


    if (potential_print)
    {
        potential_print--;
        if (order == 0) {cout << "Local chiral potential at LO: cutoff " << R0[cutoff] << " fm, lambda " << lambda[lam] << " fm." << endl;}
        else if (order == 1) {cout << "Local chiral potential at NLO: cutoff " << R0[cutoff] << " fm, lambda " << lambda[lam] << " fm." << endl;}
        else if (order == 2) {cout << "Local chiral potential at NNLO: cutoff " << R0[cutoff] << " fm, lambda " << lambda[lam] << " fm." << endl;}
    }



    parameters params = {k, kk, cutoff, S, L, LL, J, T, order, lam, channel};

    gsl_integration_workspace * w = gsl_integration_workspace_alloc (1000000);
    gsl_function VNNLO;

    VNNLO.function = &NNLOintegrand;
    VNNLO.params = &params;

    gsl_integration_qag(&VNNLO, 1e-12, 20.0, 1e-12, 1e-12, 1000000, 6, w, &sol, &error);
    gsl_integration_workspace_free (w);

    return sol;
}



/**************************************** V0 **********************************************/


double V0(double k, double kk, int pot, int S, int L, int LL, int J, int channel) {

    // pot: xyz:  x-> order: LO=0, NLO=1, N2LO=2
    //            y-> cutoff: 0.8=0, 1.0=1, 1.2=2, 0.9=3, 1.1=4
    //            z-> SFR cutoff: 800=2, 1000=3, 1200=4, 1400=5
    // Channel: nn -> -1, np -> 0, pp -> 1

    int order_in, cutoff_in, lambda_in;
    double ret;

    order_in = pot/100;
    cutoff_in = (pot-100*(pot/100))/10;
    lambda_in = pot-10*(pot/10);


    if (pot == 0) {
        strcpy(interaction, "No NN potential");
        ret=0.0;
    }
    else if ((order_in < 3) && (cutoff_in < 5) && (lambda_in < 6)) { // Local potential
        strcpy(interaction,"Local potential");
        ret=1.0/hbarc/hbarc*Vlocal(k,kk,S,L,LL,J,order_in,cutoff_in,lambda_in,channel);
    }
    else {
        unknown_potential = 1;
        ret=0.0;
    }
    return ret;
}

int main(){
    printf("exporting LEC values:\n");
    for(int order=0; order<3; order++){
        // printf("order: %d\n", order);
       for(int cutoff=0; cutoff<5; cutoff++){
            if(order==1 && cutoff==0) continue;  // these potentials don't exist
            // printf("cutoff: %d\n", cutoff);
            for(int lam=2; lam<6; lam++){
                if(order==2 && cutoff==0 && lam==5) continue;  // this potential doesn't exist
                int pot = order*100 + cutoff*10 + lam;
                double tmp = V0(0., 0., pot, 0, 0, 0, 0, 0);

                char filename[256];
                sprintf(filename, "./data/localGT+_lecs_order_%d_R0_%.1lf_lam_%.0lf.yaml", 
                        order, R0[cutoff], lambda[lam]*hbarc);
                printf("\t'%s'\n", filename);
                FILE *file = fopen(filename, "w");

                fprintf(file, "potId: %d\n", pot);
                fprintf(file, "order: %d\n", order);
                fprintf(file, "R0: %.1lf\n", R0[cutoff]);
                fprintf(file, "lambda: %.0lf\n", lambda[lam]*hbarc);

                fprintf(file, "CS: %.7lf\n", CS[cutoff]);
                fprintf(file, "CNN: %.7lf\n", CNN[cutoff]); // see line 260 // actually \Delta CS^(NN)
                fprintf(file, "CPP: %.7lf\n", CPP[cutoff]); // actually \Delta CS^(PP)
                fprintf(file, "CT: %.7lf\n", CT[cutoff]);

                fprintf(file, "C1: %.7lf\n", C1[cutoff]);
                fprintf(file, "C2: %.7lf\n", C2[cutoff]);
                fprintf(file, "C3: %.7lf\n", C3[cutoff]);
                fprintf(file, "C4: %.7lf\n", C4[cutoff]);
                fprintf(file, "C5: %.7lf\n", C5[cutoff]);
                fprintf(file, "C6: %.7lf\n", C6[cutoff]);
                fprintf(file, "C7: %.7lf\n", C7[cutoff]);

                fclose(file);
            }
        }
    }

    #if UNITTEST
    printf("testing affine decomposition\n");
    Channel channel_arr[] = {{0,0,0,0,0}, {1,0,0,1,0}, {1,2,2,1,0}, 
                             {1,2,0,1,0}, {1,0,2,1,0}};
    for(int order=0; order<3; order++){
        // printf("order: %d\n", order);
       for(int cutoff=0; cutoff<5; cutoff++){
            if(order==1 && cutoff==0) continue;  // these potentials don't exist
            // printf("cutoff: %d\n", cutoff);
            for(int lam=2; lam<6; lam++){
                if(order==2 && cutoff==0 && lam==5) continue;  // this potential doesn't exist
                int pot = order*100 + cutoff*10 + lam;
                double r = 1.;
                double ret[16] = {};
                for(auto & chan : channel_arr)
                        Vrlocal_affine(r, pot, &chan, ret);  // includes the unittest
                // for(int i = iVconst; i<iCMAX; i++) printf("%d: %lf\n", i, ret[i]);
            }
        }
    }
    #endif

    return EXIT_SUCCESS;
}