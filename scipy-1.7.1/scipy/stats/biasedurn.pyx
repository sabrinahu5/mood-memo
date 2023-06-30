# distutils: language = c++

DEF NPY_OLD = False

from .biasedurn cimport CFishersNCHypergeometric, StochasticLib3
cimport numpy as np
import numpy as np
from libcpp.memory cimport unique_ptr

IF not NPY_OLD:
    from numpy.random cimport bitgen_t
    from cpython.pycapsule cimport PyCapsule_GetPointer, PyCapsule_IsValid

    # If this were C code we could do this:
    #     from numpy.random.c_distributions cimport random_normal
    # But this is C++, so we need to wrap the include with an extern "C"
    # to prevent name-mangling -- this is probably a Cython bug.
    # (Note: the double curly braces are due to the use of a format
    #  string in _generate_pyx.py for simple substitutions -- they will
    #  be replaced by single curly braces in the generated pyx file)
    cdef extern from *:
        """
        extern "C" {
          #include "numpy/random/distributions.h"
        }
        """
        double random_normal(bitgen_t*, double, double) nogil

cdef class _PyFishersNCHypergeometric:
    cdef unique_ptr[CFishersNCHypergeometric] c_fnch

    def __cinit__(self, int n, int m, int N, double odds, double accuracy):
        self.c_fnch = unique_ptr[CFishersNCHypergeometric](new CFishersNCHypergeometric(n, m, N, odds, accuracy))

    def mode(self):
        return self.c_fnch.get().mode()

    def mean(self):
        return self.c_fnch.get().mean()

    def variance(self):
        return self.c_fnch.get().variance()

    def probability(self, int x):
        return self.c_fnch.get().probability(x)

    def moments(self):
        cdef double mean, var
        self.c_fnch.get().moments(&mean, &var)
        return mean, var


cdef class _PyWalleniusNCHypergeometric:
    cdef unique_ptr[CWalleniusNCHypergeometric] c_wnch

    def __cinit__(self, int n, int m, int N, double odds, double accuracy):
        self.c_wnch = unique_ptr[CWalleniusNCHypergeometric](new CWalleniusNCHypergeometric(n, m, N, odds, accuracy))

    def mode(self):
        return self.c_wnch.get().mode()

    def mean(self):
        return self.c_wnch.get().mean()

    def variance(self):
        return self.c_wnch.get().variance()

    def probability(self, int x):
        return self.c_wnch.get().probability(x)

    def moments(self):
        cdef double mean, var
        self.c_wnch.get().moments(&mean, &var)
        return mean, var


IF NPY_OLD:
    cdef object _glob_rng
    cdef double next_double() nogil:
        with gil:
            return _glob_rng.uniform()
    cdef double next_normal(const double m, const double s) nogil:
        with gil:
            return _glob_rng.normal(m, s)
ELSE:
    cdef bitgen_t* _glob_rng
    cdef double next_double() nogil:
        global _glob_rng
        return _glob_rng.next_double(_glob_rng.state)
    cdef double next_normal(const double m, const double s) nogil:
        global _glob_rng
        return random_normal(_glob_rng, m, s)

    cdef object make_rng(random_state=None):
        # get a bit_generator object
        if random_state is None or isinstance(random_state, int):
            bg = np.random.RandomState(random_state)._bit_generator
        elif isinstance(random_state, np.random.RandomState):
            bg = random_state._bit_generator
        elif isinstance(random_state, np.random.Generator):
            bg = random_state.bit_generator
        else:
            raise ValueError('random_state is not one of None, int, RandomState, Generator')
        capsule = bg.capsule
        return capsule


cdef class _PyStochasticLib3:
    cdef unique_ptr[StochasticLib3] c_sl3
    IF not NPY_OLD:
        cdef object capsule
        cdef bitgen_t* bit_generator

    def __cinit__(self):
        self.c_sl3 = unique_ptr[StochasticLib3](new StochasticLib3(0))
        self.c_sl3.get().next_double = &next_double
        self.c_sl3.get().next_normal = &next_normal

    def Random(self):
        return self.c_sl3.get().Random()

    def SetAccuracy(self, double accur):
        return self.c_sl3.get().SetAccuracy(accur)

    IF NPY_OLD:
        cdef void HandleRng(self, random_state=None):
            # old numpy
            if random_state is None or isinstance(random_state, int):
                rng = np.random.RandomState(random_state)
            elif isinstance(random_state, np.random.RandomState):
                rng = random_state
            elif hasattr(np.random, 'Generator') and isinstance(random_state, np.random.Generator):
                # if we are building in a compatibility mode, we may need
                # to resolve newer numpy Generator objects
                rng = random_state
            else:
                raise TypeError('random_state is not one of None, int, RandomState, or Generator')
            global _glob_rng
            _glob_rng = rng
    ELSE:
        cdef void HandleRng(self, random_state=None):
            self.capsule = make_rng(random_state)

            # get the bitgen_t pointer
            cdef const char *capsule_name = "BitGenerator"
            if not PyCapsule_IsValid(self.capsule, capsule_name):
                raise ValueError("Invalid pointer to anon_func_state")
            self.bit_generator = <bitgen_t *> PyCapsule_GetPointer(self.capsule, capsule_name)
            global _glob_rng
            _glob_rng = self.bit_generator

    def rvs_fisher(self, int n, int m, int N, double odds, int size, random_state=None):
        # handle random state
        self.HandleRng(random_state)

        # call for each
        rvs = np.empty(size, dtype=np.float64)
        for ii in range(size):
            rvs[ii] = self.c_sl3.get().FishersNCHyp(n, m, N, odds)
        return rvs

    def rvs_wallenius(self, int n, int m, int N, double odds, int size, random_state=None):
        # handle random state
        self.HandleRng(random_state)

        # call for each
        rvs = np.empty(size, dtype=np.float64)
        for ii in range(size):
            rvs[ii] = self.c_sl3.get().WalleniusNCHyp(n, m, N, odds)
        return rvs

    def FishersNCHyp(self, int n, int m, int N, double odds):
        self.HandleRng(None)  # get default rng
        return self.c_sl3.get().FishersNCHyp(n, m, N, odds)

    def WalleniusNCHyp(self, int n, int m, int N, double odds):
        self.HandleRng(None)  # get default rng
        return self.c_sl3.get().WalleniusNCHyp(n, m, N, odds)