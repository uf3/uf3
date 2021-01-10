import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib import axes

from scipy import stats
from scipy import interpolate



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
                    subset_threshold=10000,
                    cmap=None,
                    metrics=True,
                    text_size=8,
                    units=None,
                    labels=True,
                    label_size=8,
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
        cmap = cm.viridis
    x = np.array(references)
    y = np.array(predictions)
    # Compute metrics, e.g. RMSE, before selecting random subset.
    residuals = np.subtract(y, x)
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals ** 2))
    max_over = np.max(residuals)
    max_under = np.min(residuals)
    # Randomly select subset for large datasets
    x, y = get_subsets(subset_threshold, x, y)
    # Scatter, colored by log density
    x, y, z = density_estimation(x, y)
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


def density_estimation(x, y):
    """Estimate with gaussian kernel density method. Sort by log-density."""
    xy_stack = np.vstack([x, y])
    z = stats.gaussian_kde(xy_stack)(xy_stack)
    z_sort = np.argsort(z)
    x = x[z_sort]
    y = y[z_sort]
    z = z[z_sort]
    z = np.log(z - np.min(z) + 1)  # ensure valid log domain
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
                      knots,
                      ax=None,
                      r_min=None,
                      r_max=None,
                      s_min=-1,
                      s_max=2,
                      color='gray',
                      **kwargs):
    if r_min is None:
        r_min = knots[0]
    if r_max is None:
        r_max = knots[-1]
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    colors = cm.rainbow(np.linspace(0, 1, len(coefficients)))
    x_plot = np.linspace(r_min, r_max, 1000)
    for i, c in enumerate(coefficients):
        kn = knots[i:i + 5]
        kno = np.concatenate([np.repeat(kn[0], 3),
                              kn,
                              np.repeat(kn[-1], 3)])
        bs = interpolate.BSpline(kno,
                                 np.array([0, 0, 0, c, 0, 0, 0]),
                                 3,
                                 extrapolate=False)
        y_plot = bs(x_plot)
        ax.plot(x_plot, y_plot, color=colors[i], **kwargs)
        y_plot[np.isnan(y_plot)] = 0
    bs_t = interpolate.BSpline(knots,
                               coefficients,
                               3,
                               extrapolate=False)
    ax.plot(x_plot, bs_t(x_plot), c=color, **kwargs)
    ax.set_xlim(r_min - 0.1, r_max + 0.1)
    ax.set_ylim(s_min, s_max)
    return fig, ax