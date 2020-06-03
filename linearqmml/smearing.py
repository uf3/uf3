import numpy as np


def triangle_pdf(x, mu, sigma):
    pdf = [0] * len(x)  # preallocate to prepare for cython
    for i in range(len(x)):
        if (x[i] < (mu - sigma)):
            pdf[i] = 0
        elif (x[i] > (mu + sigma)):
            pdf[i] = 0
        elif (x[i] < mu):
            pdf[i] = 1/sigma + (x[i] - mu)/(sigma**2)
        else:
            pdf[i] = 1/sigma - (x[i] - mu)/(sigma**2)
    return pdf


def triangle_cdf(x, mu, sigma):
    """analytical integral of triangle function"""
    cdf = [0] * len(x)  # preallocate to prepare for cython
    for i in range(len(x)):
        if (x[i] < (mu - sigma)):
            cdf[i] = 0
        elif (x[i] > (mu + sigma)):
            cdf[i] = 1
        elif (x[i] < mu):
            cdf[i] = (x[i] - mu + sigma) ** 2 / (2 * sigma ** 2)
        else:
            cdf[i] = 1 - (x[i] - mu - sigma) ** 2 / (2 * sigma ** 2)
    return cdf


class RealspaceHandler:
    def __init__(self, sigma_factor, lower_bound, upper_bound,
                 resolutions):
        self.sigma_factor = sigma_factor
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.resolutions = resolutions
        self.cdf = triangle_cdf  # TODO: Options
        self.bin_clip = np.array(resolutions) - 1
        self.bin_widths = np.divide(np.subtract(upper_bound, lower_bound),
                                    resolutions)
        self.sigma = np.multiply(self.sigma_factor, self.bin_widths)
        self.bin_ranges = [np.arange(n_bins) / n_bins * (r_max - r_min)
                           for n_bins, r_min, r_max
                           in zip(resolutions, lower_bound, upper_bound)]

    def smear(self, mu, sigma, bin_range, idx_lower, idx_upper):
        try:
            bin_edges = bin_range[idx_lower:idx_upper + 1]
            cdfs = self.cdf(bin_edges, mu, sigma)
            bin_sums = np.subtract(cdfs[1:], cdfs[:-1])
        except ValueError:
            return np.zeros(idx_upper - idx_lower + 1)
        return bin_sums

    def smear_nd(self, x):
        x_relative = np.subtract(x, self.lower_bound)  # use the box origin
        delta = self.sigma  # how far from x to compute the integrals
        local_lower = np.floor(np.divide(x_relative - delta, self.bin_widths))
        # index of lower bound of bins across which to integrate
        local_upper = np.ceil(np.divide(x_relative + delta, self.bin_widths))
        # index of upper bound of bins across which to integrate
        local_lower = np.clip(local_lower, 0, self.bin_clip).astype(int)
        local_upper = np.clip(local_upper, 0, self.bin_clip).astype(int)

        components = [self.smear(xi, sigma, bin_range, idx_lower, idx_upper)
                      for xi, sigma, bin_range, idx_lower, idx_upper
                      in zip(x_relative, self.sigma, self.bin_ranges,
                             local_lower, local_upper)]
        # compute 1D smeared vector for each dimension
        if len(components) == 1:
            grid = components[0]
            return grid, int(local_lower), int(local_upper)
        elif len(components) == 2:
            grid = np.einsum('i,j->ij', components[0], components[1])
        elif len(components) == 3:
            grid = np.einsum('i,j,k->ijk',
                             components[0],
                             components[1],
                             components[2])
        else:
            raise ValueError
        return grid, local_lower, local_upper

    def describe(self, x_list):
        full_grid = np.zeros(self.resolutions)
        for x in x_list:
            grid, local_lower, local_upper = self.smear_nd(x)
            # TODO: generalize dimensions
            if len(self.resolutions) == 1:
                full_grid[local_lower:local_upper] += grid
            elif len(self.resolutions) == 2:
                x_min, y_min = local_lower
                x_max, y_max = local_upper
                full_grid[x_min:x_max, y_min:y_max] += grid
            elif len(self.resolutions) == 3:
                x_min, y_min, z_min = local_lower
                x_max, y_max, z_max = local_upper
                full_grid[x_min:x_max, y_min:y_max, z_min:z_max] += grid
            else:
                raise ValueError
        return full_grid


def distance_matrix_periodic(x, l):
    """generates distance matrix, assuming orthogonal lattice vectors
    and a large enough box that no atom sees itself"""
    N = len(x)
    dm = np.zeros((N,N))
    for i, row in enumerate(x):
        centered_row = np.mod(np.subtract(x, x[i]) + l/2, l)
        for j, column in enumerate(x):
            dm[i, j] = np.sqrt(np.sum(np.subtract(centered_row[i],
                                                  centered_row[j])**2))
    return dm


# def smear_nd(x, sigma, gmin, g_width, bin_ranges, g_clip):
#     """Generate a triangle distribution in n-D and bin with integration"""
#     D = len(x)
#     x_rel = np.subtract(x, gmin)  # use the box origin
#     delta = 3 * sigma  # how far from x to compute the integrals
#     dist_lower = np.floor(np.divide(x_rel - delta, g_width)).astype(int)
#     # index of lower bound of bins across which to integrate
#     dist_upper = np.ceil(np.divide(x_rel + delta, g_width)).astype(int)
#     dist_lower = np.clip(dist_lower, 0, g_clip)
#     dist_upper = np.clip(dist_upper, 0, g_clip)
#     # index of upper bound of bins across which to integrate
#     components = [smear_1d(xi, bin_range, idx_lower, idx_upper)
#                   for xi, bin_range, idx_lower, idx_upper,
#                   in zip(x_rel, sigma, bin_ranges, dist_lower, dist_upper)]
#     # compute 1D smeared vector for each dimension
#     # grid = get_tiled(components)
#     if len(components) == 1:
#         grid = components[0]
#     if len(components) == 2:
#         grid = np.einsum('i,j->ij', components[0], components[1])
#     elif len(components) == 3:
#         grid = np.einsum('i,j,k->ijk',
#                          components[0],
#                          components[1],
#                          components[2])
#     return grid, dist_lower, dist_upper
#
#
# def accumulate_smearing(x_list, sigma, gmin,  g_width, bin_ranges, g_res):
#     """Given a list of peak positions, generate and accumulate triangle
#     distributions on a grid"""
#     full_grid = np.zeros(g_res)
#     g_clip = np.array(g_res)-1
#     for x in x_list:
#         grid, cloud_lower, cloud_upper = smear_nd(x,
#                                                   sigma,
#                                                   gmin,
#                                                   g_width,
#                                                   bin_ranges,
#                                                   g_clip)
#         if len(g_res) == 1:
#             full_grid[cloud_lower:cloud_upper] += grid
#         if len(g_res) == 2:
#             x_min, y_min = cloud_lower
#             x_max, y_max = cloud_upper
#             full_grid[x_min:x_max, y_min:y_max] += grid
#         elif len(g_res) == 3:
#             x_min, y_min, z_min = cloud_lower
#             x_max, y_max, z_max = cloud_upper
#             full_grid[x_min:x_max, y_min:y_max, z_min:z_max] += grid
#     return full_grid