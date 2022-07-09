import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import axes
from scipy import stats
from scipy import interpolate
from scipy import linalg
from uf3.util import cubehelix


def round_lims(values, round_factor=0.5):
    """
    Identify rounded minimum and maximum based on appropriate power of 10
        and round_factor.

        round_place = 10 ** ceil( log10((max-min))-1 )
        Minimum = (floor(min / round_place / round_factor)
                   * round_place * round_factor)
        Maximum = (ceil(max / round_place / round_factor)
                   * round_place * round_factor)

        E.g. [10, 39, 43] yields (10, 50) with round_factor = 1 (nearest 10)
             [10, 39, 43] yields (0, 100) with round_factor = 10 (nearest 100)
             [10, 39, 43] yields (0, 45) with round_factor = 0.5 (nearest 5)
    Args:
        values (np.ndarray, list): vector of values of interest.
        round_factor (float): multiplicative factor for rounding power
            (Default = 0.5).

    Returns:
        lims: tuple of (rounded minimum, rounded maximum)

    """
    min_val = np.min(values)
    max_val = np.max(values)
    round_place = 10 ** np.ceil(np.log10(np.ptp([min_val, max_val])) - 1)
    rounded_min = (np.floor(min_val / round_place / round_factor)
                   * round_place * round_factor)
    rounded_max = (np.ceil(max_val / round_place / round_factor)
                   * round_place * round_factor)
    lims = (rounded_min, rounded_max)
    tick_factor = round_place * round_factor
    return lims, tick_factor


def density_scatter(references,
                    predictions,
                    ax=None,
                    loglog=False,
                    lims=None,
                    lim_factor=0.5,
                    subset_threshold=1000,
                    cmap=None,
                    metrics=True,
                    text_size=10,
                    units=None,
                    labels=True,
                    label_size=10,
                    **scatter_kwargs):
    """
    Plot regression performance with a scatter plot of predictions vs.
        references, colored by log-density of points. Optionally display
        mean-absolute error, root-mean-square error, minimum residual,
        and maximum residual.

    Args:
        references (list, np.ndarray): Vector of Y-axis values.
        predictions (list, np.ndarray): Vector of X-axis values.
        ax (axes.Axes): Optional handle for existing matplotlib axis object
        loglog (bool): whether to plot on a log-log scale.
        lims (tuple): lower and upper bounds for axis limits.
        lim_factor (float): tuning factor for automatically determining limits.
        subset_threshold (int): maximum number of points to plot.
            If exceeded, subset will be selected randomly.
        cmap (matplotlib.colors.LinearSegmentedColormap): color map.
        metrics (bool): plot text with metrics e.g. root-mean-square-error.
        text_size (int): fontsize for metrics text.
        units (str): units for axis labels.
        labels (bool): add axis labels.
        label_size (int): fontsize for axis and tick labels.
        **scatter_kwargs: keyword arguments for plt.scatter function.

    Returns:
        fig & ax: matplotlib figure and axis.
    """
    if ax is None:
        fig, ax = plt.subplots()
        fig_tuple = (fig, ax)
    else:
        fig_tuple = (None, None)
    if 's' not in scatter_kwargs.keys():
        scatter_kwargs['s'] = 1  # default marker size
    if cmap is None:
        cmap = cubehelix.c_rainbow
    x = np.array(references)
    y = np.array(predictions)
    # Compute metrics, e.g. RMSE, before selecting random subset.
    residuals = np.subtract(y, x)
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals ** 2))
    max_over = np.max(residuals)
    max_under = np.min(residuals)
    # Randomly select subset for large datasets
    x_subset, y_subset = get_subsets(subset_threshold, x, y)
    # Scatter, colored by log density
    try:
        x, y, z = density_estimation(x_subset, y_subset, x, y)
    except linalg.LinAlgError:
        z = np.ones(len(y))
    ax.scatter(x, y, c=z, cmap=cmap, **scatter_kwargs)
    # Axis scale and limits
    ax.axis('square')
    if loglog is True:
        ax.set_xscale('log')
        ax.set_yscale('log')
        if lims is None:
            lims = ax.get_xlim()
    else:
        if lims is None:
            lims, tick_factor = round_lims(np.concatenate([x, y]),
                                           round_factor=lim_factor)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.plot(lims, lims, color='lightgray', linestyle='--', linewidth=0.5)
    # Error Metrics
    if metrics is True:
        error_text = 'RMSE = {0:.3f}'.format(rmse)
        error_text += '\nMAE = {0:.3f}'.format(mae)
        res_text = 'Max Res. = {0:.3f}'.format(max_over)
        res_text += '\nMin Res. = {0:.3f}'.format(max_under)
        ax.text(0.02, 0.98, error_text,
                ha='left', va='top',
                fontsize=text_size,
                transform=ax.transAxes)
        ax.text(0.98, 0.02, res_text,
                ha='right', va='bottom',
                fontsize=text_size,
                transform=ax.transAxes)
    # Axis Labels
    if labels is True:
        if isinstance(units, str):
            unit_string = " " + units
            if all([c not in unit_string for c in ['[', ']', '(', ')']]):
                unit_string = ' [{}]'.format(units)
        else:
            unit_string = ""
        ax.set_ylabel('Predicted' + unit_string, fontsize=label_size)
        ax.set_xlabel('Reference' + unit_string, fontsize=label_size)
    ax.tick_params(axis='both', labelsize=label_size)
    return fig_tuple


def density_estimation(x_subset, y_subset, x, y):
    """Estimate with gaussian kernel density method. Sort by log-density."""
    xy_subset = np.vstack([x_subset, y_subset])
    xy_stack = np.vstack([x, y])
    z = stats.gaussian_kde(xy_subset)(xy_stack)
    z_sort = np.argsort(z)
    x = x[z_sort]
    y = y[z_sort]
    z = z[z_sort]
    z = np.log10(z - np.min(z) + 1)  # ensure valid log domain
    return x, y, z


def get_subsets(subset_threshold, *args):
    """

    Args:
        subset_threshold: minimum threshold of points to sample.
        *args: lists or vectors or arrays of the same size to slice.

    Returns:
        *new_args: subsets of input vectors.
    """
    x = args[0]
    n_points = len(x)
    if n_points > subset_threshold:
        subset_indices = np.random.choice(np.arange(len(x)),
                                          int(subset_threshold),
                                          replace=False)
        new_args = [np.take(arg, subset_indices, axis=0)
                    for arg in args]
        return new_args
    else:
        return args


def visualize_splines(coefficients,
                      knot_sequence,
                      ax=None,
                      cmap=None,
                      show_components=True,
                      show_total=True):
    r_min = knot_sequence[0]
    r_max = knot_sequence[-1]
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    if cmap is None:
        cmap = cubehelix.c_rainbow
    colors = cmap(np.linspace(0, 1, len(coefficients)))
    x_plot = np.linspace(r_min, r_max, 1000)
    basis_components = []

    for i, c in enumerate(coefficients):
        kn = knot_sequence[i:i + 5]
        kno = np.concatenate([np.repeat(kn[0], 3),
                              kn,
                              np.repeat(kn[-1], 3)])
        bs = interpolate.BSpline(kno,
                                 np.array([0, 0, 0, c, 0, 0, 0]),
                                 3,
                                 extrapolate=False)
        y_plot = bs(x_plot)

        if show_components:
            ax.plot(x_plot,
                    y_plot,
                    color=colors[i],
                    linewidth=1)
        y_plot[np.isnan(y_plot)] = 0
        basis_components.append(y_plot)
    y_total = np.sum(basis_components, axis=0)
    s_min = np.min(y_total[~np.isnan(y_total)])
    s_max = np.max(y_total[~np.isnan(y_total)])
    if show_total:
        ax.plot(x_plot,
                y_total,
                c='k',
                linewidth=2)
    ax.set_xlim(r_min, r_max)
    ax.set_ylim(s_min, s_max)
    ax.set_xlabel("r")
    ax.set_ylabel("B(r)")
    return fig, ax


def visualize_basis_functions(coefficients,
                              knot_sequence,
                              ax=None,
                              cmap=None):
    r_min = knot_sequence[0]
    r_max = knot_sequence[-1]
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    if cmap is None:
        cmap = cubehelix.c_rainbow
    colors = cmap(np.linspace(0, 1, len(coefficients)))
    x_plot = np.linspace(r_min, r_max, 1000)
    basis_components = []

    for i, c in enumerate(coefficients):
        kn = knot_sequence[i:i + 5]
        kno = np.concatenate([np.repeat(kn[0], 3),
                              kn,
                              np.repeat(kn[-1], 3)])
        bs = interpolate.BSpline(kno,
                                 np.array([0, 0, 0, c, 0, 0, 0]),
                                 3,
                                 extrapolate=False)
        y_plot = bs(x_plot)

        ax.plot(x_plot,
                y_plot,
                color=colors[i],
                linewidth=1)
        y_plot[np.isnan(y_plot)] = 0
        basis_components.append(y_plot)
    y_total = np.sum(basis_components, axis=0)
    s_min = np.min(y_total[~np.isnan(y_total)])
    s_max = np.max(y_total[~np.isnan(y_total)])
    ax.set_xlim(r_min, r_max)
    ax.set_ylim(s_min, s_max)
    ax.set_xlabel("r")
    ax.set_ylabel("B(r)")
    return fig, ax


def visualize_pair_potential(coefficients,
                             knot_sequence,
                             ax=None,
                             **kwargs):
    r_min = knot_sequence[0]
    r_max = knot_sequence[-1]
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    if "linewidth" not in kwargs:
        kwargs["linewidth"] = 2
    if "color" not in kwargs:
        kwargs["color"] = "black"

    x_plot = np.linspace(r_min, r_max, 1000)
    basis_components = []
    for i, c in enumerate(coefficients):
        kn = knot_sequence[i:i + 5]
        kno = np.concatenate([np.repeat(kn[0], 3),
                              kn,
                              np.repeat(kn[-1], 3)])
        bs = interpolate.BSpline(kno,
                                 np.array([0, 0, 0, c, 0, 0, 0]),
                                 3,
                                 extrapolate=False)
        y_plot = bs(x_plot)
        y_plot[np.isnan(y_plot)] = 0
        basis_components.append(y_plot)
    y_total = np.sum(basis_components, axis=0)
    s_min = np.min(y_total[~np.isnan(y_total)])
    s_max = np.max(y_total[~np.isnan(y_total)])
    ax.plot(x_plot,
            y_total,
            **kwargs)
    ax.set_xlim(r_min, r_max)
    ax.set_ylim(s_min, s_max)
    ax.set_xlabel("r")
    ax.set_ylabel("B(r)")
    return fig, ax


def plot_pair_distributions(analysis,
                            pair_order=None,
                            x_max=None,
                            y_max=2.0,
                            show_cutoffs=False,
                            figsize=(3.5, 3),
                            dpi=100):
    frequencies = analysis["rdfs"]
    bin_edges = analysis["bin_edges"]
    valleys = analysis["valleys"]
    if pair_order is None:
        pair_order = list(frequencies.keys())
    if x_max is None:
        x_max = bin_edges[-1]
    bar_width = bin_edges[1] - bin_edges[0]
    canvases = []
    for i, pair in enumerate(pair_order):
        fig, ax = plt.subplots(figsize=figsize,
                               dpi=dpi)
        ax.set_title(" - ".join(pair))
        ax.set_xlim(0, x_max)
        if y_max is None:
            vector = frequencies[pair]
            vector = vector[np.nonzero(vector)]
            y_lim = np.mean(vector) * 2
        else:
            y_lim = y_max
        ax.set_ylim(0, y_lim)
        ax.bar(bin_edges[:-1],
               frequencies[pair],
               width=bar_width)
        ax.plot([0, x_max],
                [1.0, 1.0],
                linestyle='--',
                color='k')
        if show_cutoffs:
            ax.vlines(valleys.get(pair, []),
                      0,
                      y_lim,
                      color="orange",
                      linestyle=":")
        ax.set_xlabel("Pair distance (angstroms)")
        ax.set_ylabel("Normalized Frequency")
        canvases.append((fig, ax))
    return canvases
