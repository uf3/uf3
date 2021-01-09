import numpy as np


def triangle_pdf(x, mu, sigma):
    """triangle function"""
    pdf = [0] * len(x)  # preallocate to prepare for cython
    for i in range(len(x)):
        if (x[i] < (mu - sigma)):
            pdf[i] = 0
        elif (x[i] > (mu + sigma)):
            pdf[i] = 0
        elif (x[i] < mu):
            pdf[i] = 1 / sigma + (x[i] - mu) / (sigma ** 2)
        else:
            pdf[i] = 1 / sigma - (x[i] - mu) / (sigma ** 2)
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


def get_bin_edges(n_bins, r_min, r_max, power):
    """
    Generate (n_bins + 1) bin edges spanning r_min to r_max.

    Args:
        n_bins (int): number of bins.
        r_min (float): minimum value (left-most bin edge).
        r_max (float): maximum value (right-most bin edge).
        power (int): power scaling factor for weighting bin widths.

    Returns:
        bin_edges (list): list of n_bins+1 bin edges.
    """
    def u_func(x, p=1):
        return x ** p
    assert power in [1, -1, -2]
    # TODO: fractional powers & warn about threshold based on constraints

    delta = abs(u_func(r_min, power) - u_func(r_max, power)) / (n_bins)
    transformed_edges = np.arange(-1, n_bins + 1) * delta

    if power < 0:
        transformed_edges = transformed_edges + u_func(r_max,
                                                       power) + delta
        assert not any(transformed_edges < 0), "Bins diverge to infinity. " \
                                               "Please increase number of " \
                                               "bins, decrease range, " \
                                               "or decrease power magnitude. "
        bin_edges = transformed_edges ** (1 / power)
        bin_edges = np.sort(bin_edges[:-1])
    else:
        bin_edges = transformed_edges[1:]
        bin_edges += u_func(r_min, power)
    return bin_edges


class RealspaceHandler:
    """Class for generating real-space representations."""
    def __init__(self, sigma_factor, lower_bound, upper_bound,
                 resolutions, transform_power=1):
        """
        Args:
            sigma_factor (float): smearing parameter to be multiplied by the
                characteristic bin width for each dimension.
            lower_bound (list): list of lower bound (floats) per dimension.
            upper_bound (list): list of upper bound (floats) per dimension.
            resolutions (list): list of number of bins (int) per dimension.
            transform_power (int): scaling power for weighting bin widths.
        """
        self.sigma_factor = sigma_factor
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.resolutions = resolutions
        self.cdf = triangle_cdf  # TODO: Options
        self.bin_clip = np.array(resolutions)
        self.avg_resoln = np.divide(np.subtract(upper_bound, lower_bound),
                                    resolutions)
        self.sigma = np.multiply(self.sigma_factor, self.avg_resoln)

        self.transform_power = transform_power
        self.bin_edges = [get_bin_edges(n_bins,
                                        r_min,
                                        r_max,
                                        self.transform_power)
                          for n_bins, r_min, r_max
                          in zip(resolutions, lower_bound, upper_bound)]
        # TODO: remove depracated linear-spacing lines
        # self.bin_edges = [np.arange(n_bins) / n_bins * (r_max - r_min)
        #                   for n_bins, r_min, r_max
        #                   in zip(resolutions, lower_bound, upper_bound)]

    def smear(self, mu, sigma, bin_edges, idx_lower, idx_upper):
        """
        Smear (broaden) distribution centered on mu with width sigma and
            integrate in intervals specified by bin_edges and idx range.

        Args:
            mu (float): center of distribution.
            sigma (float): width of broadened distribution.
            bin_edges (list): list of bin edges (n_bins + 1)
            idx_lower (int): index of left-most bin edge in effective range
            idx_upper (int): index of right-most bin edge in effective range

        Returns:
            bin_sums (list): integrated value of distribution per bin.
        """
        try:
            bin_edges = bin_edges[idx_lower:idx_upper + 1]
            cdfs = self.cdf(bin_edges, mu, sigma)
            bin_sums = np.subtract(cdfs[1:], cdfs[:-1])
        except KeyboardInterrupt:
            return np.zeros(idx_upper - idx_lower + 1)
        return bin_sums

    def smear_nd(self, x):
        """
        Smear (broaden) distribution with support for multiple dimensions.

        Args:
            x (tuple): coordinate.

        Returns:
            grid (np.ndarray): local subgrid with integrated values
                from distribution centered on x.
            local_lower (list): lower indices of subgrid on grid
            local_upper (list): upper indices of subgrid on grid
        """
        local_lower = [np.argmin((bin_edges - (xi - delta)) ** 2)
                       for xi, bin_edges, delta
                       in zip(x, self.bin_edges, self.sigma)]
        # index of lower bound of bins across which to integrate
        local_upper = [np.argmin((bin_edges - (xi + delta)) ** 2)
                       for xi, bin_edges, delta
                       in zip(x, self.bin_edges, self.sigma)]
        for dim in range(len(x)):
            if local_lower[dim] == local_upper[dim]:
                # TODO: better handling of case where entire dist in 1 bin
                local_lower[dim] -= 1
                local_upper[dim] += 1
        # index of upper bound of bins across which to integrate
        local_lower = np.clip(local_lower, 0, self.bin_clip).astype(int)
        local_upper = np.clip(local_upper, 0, self.bin_clip).astype(int)

        components = [self.smear(xi, sigma, bin_edges,
                                 idx_lower, idx_upper)
                      for xi, sigma,
                          bin_edges,
                          idx_lower, idx_upper
                      in zip(x, self.sigma,
                             self.bin_edges,
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

    def describe_1d(self, sample_geometry):
        """
        Generate 1D representation from pairwise (2-body) interactions
            in sample_geometry.

        Args:
            sample_geometry: ase.Atoms object.

        Returns:
            full_grid (np.ndarray): 1D representation vector.
        """
        r_min = self.lower_bound[0]
        r_max = self.upper_bound[0]
        skin = self.sigma[0]
        positions = sample_geometry.get_positions().astype(np.float64)
        cell = sample_geometry.cell.diagonal().astype(np.float64)
        sample_distance_matrix = distance_matrix_periodic(positions, cell)
        matrix_mask = np.logical_and(sample_distance_matrix > r_min,
                                     sample_distance_matrix < r_max + skin)
        sample_distances = sample_distance_matrix[matrix_mask]
        if len(np.shape(sample_distances)) == 1:
            sample_distances = sample_distances[:, np.newaxis]
        full_grid = np.zeros(self.resolutions)
        for x in sample_distances:
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


def distance_matrix_periodic(x, lattice_vectors):
    """
    Generates distance matrix, assuming orthogonal lattice vectors
    and a large enough box that no atom sees itself
    
    Args:
        x (list): list of N coordinates.
        lattice_vectors (list): list of lattice vectors (3x3).

    Returns:
        distance_matrix (np.ndarray): N x N pairwise distance matrix.
    """

    N = len(x)
    distance_matrix = np.zeros((N, N))
    for i, row in enumerate(x):
        centered_row = np.mod(np.subtract(x, x[i]) + lattice_vectors / 2, 
                              lattice_vectors)
        for j, column in enumerate(x):
            difference = np.subtract(centered_row[i], centered_row[j]) ** 2
            distance_matrix[i, j] = np.sqrt(np.sum(difference))
    return distance_matrix

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
