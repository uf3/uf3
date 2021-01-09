import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import gaussian_kde
from linearqmml.utils import get_morse_potential


def plot_distance_histogram(distances, n_bins=50, ax=None):
    """
    Histogram of pairwise distances, scaled by 1/r^2. The histogram converges
    to the number density at larger distances.
    
    Args:
        distances (list): flattened list of distance observations.
        n_bins (int): Number of bins for histogram. Defaults to 50.
        ax: optional matplotlib axis object on which to plot.

    Returns:
        fig & ax: new matplotlib figure & axis if ax is not specified.
    """
    if ax is None:
        return_ax = True
        fig, ax = plt.subplots(figsize=(8, 4.5))
    else:
        return_ax = False
    histogram, bin_edges = np.histogram(np.concatenate(distances),
                                        bins=n_bins)
    histogram_r2 = [bin_count / r ** 2
                    for bin_count, r in zip(histogram, bin_edges)]
    ax.bar(bin_edges[1:], histogram_r2, width=bin_edges[1] - bin_edges[2])
    if return_ax:
        return fig, ax


def scatter_rmse(predictions, references, factor=1,
                 subset_fraction=1, ax=None, **kwargs):
    """
    Scatter plot of predictions vs. references, colored by density.

    Args:
        predictions (list): List of floats.
        references (list): List of floats.
        factor (float): Scaling factor, e.g. for unit conversion.
        subset_fraction (float): Fraction of points to include (randomly).
        ax: Optional handle for existing matplotlib axis object
        **kwargs: Optional keyword arguments for scatter function.

    Returns:
        fig & ax: new matplotlib figure and axis if ax is not specified.
        rmse_text (str): Summary of root mean square error.
    """
    if ax is None:
        return_ax = True
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        return_ax = False
    x = np.array(predictions) * factor
    y = np.array(references) * factor
    rmse = np.sqrt(np.mean(np.subtract(x, y) ** 2))
    z = np.vstack([x, y])
    n_pts = len(x)
    subset_fraction = np.random.choice(np.arange(n_pts),
                                       int(np.ceil(n_pts * subset_fraction)),
                                       replace=False)
    z_subset = np.take(z, subset_fraction, axis=1)
    z = gaussian_kde(z_subset)(z)
    zsort = np.argsort(z)
    z = z[zsort] / np.max(z)
    x = x[zsort]
    y = y[zsort]
    z_scaled = 0.75 * z + 0.25
    colors = np.array([cm.viridis(zi) for zi in z])
    colors[:, -1] = z_scaled
    ax.scatter(x, y, c=colors, **kwargs)
    xmin, xmax = ax.get_xlim()
    xr = [np.floor(xmin), np.ceil(xmax)]
    ymin, ymax = ax.get_ylim()
    yr = [np.floor(ymin), np.ceil(ymax)]
    ax.plot(xr, xr, 'k--')
    ax.set_xlim(xr)
    ax.set_ylim(yr)
    rmse_text = 'RMSE = {0:.4f} meV/atom'.format(rmse)
    print(rmse / np.ptp(xr) * 100, '% error')
    ax.axis('equal')
    ax.set_ylabel('True Energy (meV/atom)')
    ax.set_xlabel('Predicted Energy (meV/atom)')
    if return_ax:
        return fig, ax, rmse_text
    else:
        return rmse_text


def plot_representations(representations, r, ax=None):
    """
    Plot distance-based representation.

    Args:
        representations (list): list of fixed-length representations.
        r (list): list of r values corresponding to representation
        ax: optional matplotlib axis object on which to plot.

    Returns:
        fig & ax: new matplotlib figure & axis if ax is not specified.
    """
    if ax is None:
        return_ax = True
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        return_ax = False

    average = np.mean(representations, axis=0)
    for i in range(len(representations)):
        ax.plot(r, representations[i],
                color=cm.jet(i / len(representations)), alpha=0.1)
    ax.plot(r, average, color='k', linewidth=3)
    ax.set_xlim(r[0], r[-1])
    if return_ax:
        return fig, ax


def plot_morse_performance(coefficients,
                           predictions,
                           references,
                           sample_representations,
                           bin_edges,
                           figsize=(13 * 0.75, 4)):
    """
    Plot 3 panel summary of performance with morse potential.

    Args:
        coefficients (list): regression coefficients.
        predictions (list): input for scatter_rmse.
        references (list): input for scatter_rmse.
        sample_representations (list): input for plot_representations.
        bin_edges (list): list of bin edges for representation.
        figsize (tuple): optional tuple of figure width and length.

    Returns:
        fig & ax: new matplotlib figure & axis objects.
    """
    r = bin_edges[:-1]
    fig, ax = plt.subplots(1, 3, figsize=figsize)
    plot_representations(sample_representations, r, ax=ax[0])
    ax[0].set_xlabel('r ($\mathrm{\AA}$)')
    ax[0].set_ylabel('$\mathrm{g(r)}$')
    ax[0].set_title('Representations')

    morse = get_morse_potential(r)
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    ax[1].plot(r, morse, 'blue', label='Morse\nPotential', zorder=1)
    # ax[1].plot(r, coefficients, color='#FF7F00', marker='s', linestyle=':',
    #            label=r'$\beta$', zorder=0, markersize=1.5)
    ax[1].bar(bin_edges[:-1],
              coefficients,
              edgecolor='black', linewidth=1,
              align='edge',
              color='#FF7F00',
              label=r'$\beta$',
              width=bin_widths)

    ax[1].legend(loc=(0.25, 0.7))
    ax[1].set_xlim(r[0], r[-1])
    ax[1].set_ylim(-0.5, 1.5)
    ax[1].set_xlabel('r ($\mathrm{\AA}$)')
    ax[1].set_ylabel('V(r) (eV/atom)')
    ax[1].set_title('Regression Coefficients')

    rmse_text = scatter_rmse(predictions, references, 2, ax=ax[2]) + '\n'
    ax[2].set_title('Testing Error')
    ax[2].axis('square')
    fig.tight_layout()
    plot_center = np.mean(ax[2].get_xlim())
    plot_bottom, plot_top = ax[2].get_ylim()
    ax[2].text(plot_center, plot_bottom, rmse_text, ha='center', va='bottom',
               fontsize=8)
    return fig, ax
