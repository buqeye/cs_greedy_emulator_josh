cimport cpot
from cpython.pycapsule cimport *
from libc.stdlib cimport malloc, free


cdef del_Channel(object obj):
    pt = <cpot.Channel *> PyCapsule_GetPointer(obj, "Channel")
    free(<void *> pt)

def Channel(int S, int L, int LL, int J, int channel):
    cdef cpot.Channel * chan
    chan = <cpot.Channel *> malloc(sizeof(cpot.Channel))

    if chan == NULL:
        raise MemoryError("No memory to make a Point")

    chan.S = S
    chan.L = L
    chan.LL = LL
    chan.J = J
    chan.channel = channel

    return PyCapsule_New(<void *>chan, "Channel",
                         <PyCapsule_Destructor>del_Channel)



cdef del_Lecs(object obj):
   pt = <cpot.Lecs *> PyCapsule_GetPointer(obj, "Lecs")
   free(<void *> pt)

def Lecs(double CS, double CT, double C1, double C2,
    double C3, double C4, double C5, double C6, double C7,
    double CNN, double CPP):
   cdef cpot.Lecs * chan
   lecs = <cpot.Lecs *> malloc(sizeof(cpot.Lecs))

   if lecs == NULL:
       raise MemoryError("No memory to make a Point")

   lecs.CS = CS
   lecs.CT = CT
   lecs.C1 = C1
   lecs.C2 = C2
   lecs.C3 = C3
   lecs.C4 = C4
   lecs.C5 = C5
   lecs.C6 = C6
   lecs.C7 = C7
   lecs.CNN = CNN
   lecs.CPP = CPP

   return PyCapsule_New(<void *>lecs, "Lecs",
                        <PyCapsule_Destructor>del_Lecs)


def V0( k,  kk,  pot,  S,  L,  LL,  J, channel):
  return cpot.V0( k,  kk,  pot,  S,  L,  LL,  J, channel)

def Vrlocal(r, pot, chan, lecs):
  cchan = <cpot.Channel *> PyCapsule_GetPointer(chan, "Channel")
  clecs = <cpot.Lecs *> PyCapsule_GetPointer(lecs, "Lecs")
  return cpot.Vrlocal(r, pot, cchan, clecs)

import numpy as np
def Vrlocal_affine(r, pot, chan, ret):  # 'ret' is a one-dimensional numpy array
    """ 
    see https://docs.cython.org/en/latest/src/userguide/memoryviews.html#pass-data-from-a-c-function-via-pointer
    """
    cchan = <cpot.Channel *> PyCapsule_GetPointer(chan, "Channel")
    if not ret.flags['C_CONTIGUOUS']:
        ret = np.ascontiguousarray(ret)  # Makes a contiguous copy of the numpy array.
    cdef double[::1] ret_memview = ret
    cpot.Vrlocal_affine(r, pot, cchan, &ret_memview[0])
    return ret