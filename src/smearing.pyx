cimport libc.math as math
cimport cython
import numpy as np
cimport numpy as cnp

DTYPE = np.float64
ctypedef cnp.float64_t DTYPE_t

# TODO: cpdef -> cdef where appropriate

@cython.wraparound (False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cpdef cnp.ndarray[DTYPE_t, ndim=1] ctriangle_cdf(cnp.ndarray[DTYPE_t, ndim=1] x, int n, DTYPE_t mu, DTYPE_t sigma):
    """Generate cumulative distribution function"""
    cdef cnp.ndarray[DTYPE_t, ndim=1] cdf = np.zeros(n, dtype=DTYPE)
    cdef DTYPE_t value
    for i in range(n + 1):
        if (x[i] < (mu - sigma)):
            value = 0
        elif (x[i] > (mu + sigma)):
            value = 1
        elif (x[i] < mu):
            value = (x[i] - mu + sigma)**2/(2*sigma**2)
        else:
            value = 1- (x[i] - mu - sigma)**2/(2*sigma**2)
        cdf[i] = value
    return cdf

@cython.wraparound (False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cpdef list csmear(DTYPE_t mu, DTYPE_t sigma, cnp.ndarray[DTYPE_t, ndim=1] r_smear, int n_smear,
                  DTYPE_t r_max):
    cdef list bin_sums = []
    cdef int n = 0
    cdef int i
    cdef cnp.ndarray[DTYPE_t, ndim=1] cdfs = ctriangle_cdf(r_smear, n_smear, mu, sigma)
    for i in range(n_smear):
        if r_smear[i] < r_max:
            bin_sums.append(cdfs[i+1] - cdfs[i])
            n += 1
    return bin_sums

@cython.wraparound (False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cpdef cnp.ndarray[DTYPE_t, ndim=1] smearbin_1d(DTYPE_t[::1] distances, double sigma_factor, double r_min,
                        double r_max, int n_bins):
    cdef cnp.ndarray[DTYPE_t, ndim=1] description = np.zeros(n_bins, dtype=DTYPE)
    cdef int n_interactions = len(distances)
    cdef int i, idx_lower, idx_upper, n_smear
    cdef list component
    cdef DTYPE_t dist

    cdef DTYPE_t bin_width = (r_max - r_min) / n_bins
    cdef DTYPE_t sigma = sigma_factor * bin_width
    cdef DTYPE_t delta = sigma * 3

    cdef cnp.ndarray[DTYPE_t, ndim=1] r_range = np.array([bin_width * i for i in range(n_bins + 1)])
    for i in range(n_interactions):
        dist = distances[i]
        idx_lower = int(math.floor((dist - r_min - delta) / bin_width)) - 1
        idx_upper = int(math.ceil((dist - r_min + delta) / bin_width))
        if idx_lower < 0:
            idx_lower = 0
        if idx_upper < 0:
            continue
        if idx_lower > (n_bins -1):
            idx_lower = n_bins - 1
        if idx_upper > n_bins:
            idx_upper = n_bins
        n_smear = idx_upper - idx_lower

        component = csmear(dist - r_min, sigma, r_range[idx_lower:idx_upper+1],
                           n_smear, r_max)
        n = len(component)
        for i in range(n):
            description[idx_lower + i] += component[i]
    return description

@cython.wraparound (False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cpdef cnp.ndarray[DTYPE_t, ndim=2] distance_matrix_periodic(cnp.ndarray[DTYPE_t, ndim=2] x, cnp.ndarray[DTYPE_t, ndim=1] boundaries):
    cdef int N = len(x)
    cdef cnp.ndarray[DTYPE_t, ndim=2] distances = np.zeros((N, N), dtype=DTYPE)
    cdef list row_translated, delta
    cdef int idx_center, idx_neighbor
    cdef DTYPE_t delta_sum
    for idx_center in range(N):
        row_translated = [[(x[idx_point][idx_dim] - x[idx_center][idx_dim]
                            + boundaries[idx_dim] / 2) % boundaries[idx_dim]
                           for idx_dim in [0,1,2]]
                          for idx_point in range(N)]

        for idx_neighbor in range(N):
            delta = [(row_translated[idx_center][idx_dim]
                     - row_translated[idx_neighbor][idx_dim]) **2
                     for idx_dim in [0,1,2]]
            delta_sum = delta[0] + delta[1] + delta[2]
            distances[idx_center][idx_neighbor] = delta_sum ** 0.5
    return distances