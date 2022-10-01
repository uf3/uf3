import ndsplines
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker
from matplotlib.patches import ConnectionPatch
from matplotlib import cm


def plot_slices(model,
                trio,
                thetas=(45, 60, 90, 120, 180),
                slice_resolution=25,
                vmin=-0.500,
                vmax=0.500,
                vscale=0.5,
                cutoff=0.300,
                dpi=150,
                r_pad=1.0,
                fig=None,
                gs=None):
    n_main = len(thetas)

    tbc = ThreeBodyCut(model, trio, thetas=thetas)
    tbc.sample_uniformly(slice_resolution)
    r_max = np.max(tbc.knots[0]) + r_pad

    if fig is None:
        fig = plt.figure(figsize=(n_main, 2.5), dpi=dpi)
    if gs is None:
        gs = GridSpec(1, 1, figure=fig, hspace=0.15)[0]
    gs = gs.subgridspec(5,
                        n_main * 2,
                        hspace=0.0,
                        wspace=0.0,
                        height_ratios=[0.1, 0.8, 1, 1, 1.2])

    theta_set = []
    for j in range(n_main):
        ax = fig.add_subplot(gs[0, j * 2: (j + 1) * 2])
        theta_set.append(ax)
    pos_set = []
    for j in range(n_main * 2):
        ax = fig.add_subplot(gs[1, j])
        pos_set.append(ax)
    neg_set = []
    for j in range(n_main * 2):
        ax = fig.add_subplot(gs[4, j])
        neg_set.append(ax)
    slice_set = []
    for j in range(n_main):
        ax = fig.add_subplot(gs[2:4, j * 2: (j + 1) * 2])
        slice_set.append(ax)

    axes = tbc.plot_slices(vmin=vmin,
                           vmax=vmax,
                           half=False,
                           axes=slice_set,
                           cmap="RdBu_r")
    titles = []
    for ax in axes:
        text = ax.get_title()
        titles.append(text)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
        ax.xaxis.set_ticklabels([])
        if ax != axes[0]:
            ax.yaxis.set_ticklabels([])
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.set_title(None)
        ax.grid(True)
    axes[0].set_ylabel(r"$r_{ik}$ [$\mathrm{\AA}$]", )
    for j, title in enumerate(titles):
        ax = theta_set[j]
        ax.text(0.5,
                0.5,
                title,
                ha="center",
                va="bottom",
                transform=ax.transAxes)
        ax.plot([0.05, 0.95],
                [0.40, 0.40],
                color="k",
                linewidth=1,
                transform=ax.transAxes)
        ax.axis("off")
    for ax in pos_set:
        ax.axis("off")
    for ax in neg_set:
        ax.axis("off")
    for j, theta in enumerate(tbc.thetas):
        i_pos = 0
        i_neg = 0
        x_grid = tbc.mesh[0]
        y_grid = tbc.mesh[1]
        grid = tbc.values[j].copy()
        grid_pos = np.tril(grid)
        grid_neg = -np.triu(grid)
        centroids_pos = find_centroids(x_grid,
                                       y_grid,
                                       grid_pos,
                                       cutoff=cutoff)
        centroids_neg = find_centroids(x_grid,
                                       y_grid,
                                       grid_neg,
                                       cutoff=cutoff)
        if len(centroids_pos) > 0:
            centroids_pos = centroids_pos[np.argsort(centroids_pos[:, 0])]
        if len(centroids_neg) > 0:
            centroids_neg = centroids_neg[np.argsort(centroids_neg[:, 0])]
        for cx, cy, v in centroids_pos:
            ax = pos_set[(j * 2) + i_pos]
            ax_square = axes[j]
            con = plot_connections(theta,
                                   v,
                                   cx,
                                   cy,
                                   ax,
                                   ax_square,
                                   (0.0, -3.5),
                                   r_max,
                                   vscale,
                                   sign="+",
                                   )
            fig.add_artist(con)
            i_pos += 1
        for cx, cy, v in centroids_neg:
            v = -v
            ax = neg_set[(j * 2) + i_neg]
            ax_square = axes[j]
            con = plot_connections(theta,
                                   v,
                                   cx,
                                   cy,
                                   ax,
                                   ax_square,
                                   (0.0, 3.5),
                                   r_max,
                                   vscale,
                                   sign="",
                                   )
            fig.add_artist(con)
            i_neg += 1
    return fig, gs


class ThreeBodyCut:

    def __init__(self, model, interaction, thetas=(45, 60, 90, 120, 180)):
        model_dump = model.dump()
        decompressed_coefficients = model_dump["coefficients"][interaction]
        knots_set = model_dump["knots"][interaction]
        self.knots_set = [np.array(k) for k in knots_set]

        self.c_min = np.min(self.knots_set[2])
        self.c_max = np.max(self.knots_set[2])

        self.name = interaction
        self.nds = ndsplines.NDSpline(self.knots_set,
                                      decompressed_coefficients,
                                      3)
        self.knots = knots_set
        self.theta = thetas
        self.mesh = None
        self.x_plot = None
        self.y_plot = None
        self.z_plot = None
        self.thetas = thetas
        self.n_cuts = len(thetas)
        self.values = None
        self.vscale = None

    def sample_uniformly(self, n_samples):
        if isinstance(n_samples, int):
            n_samples = [n_samples, n_samples]

        sample_dims = [
            np.linspace(self.knots_set[0][0],
                        self.knots_set[0][-1],
                        n_samples[0]),
            np.linspace(self.knots_set[1][0],
                        self.knots_set[1][-1],
                        n_samples[1])
        ]

        sample_mesh = np.meshgrid(*sample_dims)

        self.x_plot = sample_mesh[0]
        self.y_plot = sample_mesh[1]
        a = sample_mesh[0]
        b = sample_mesh[1]
        self.mesh = (a, b)

        values = []
        for theta in self.thetas:
            theta = np.deg2rad(theta)
            c = np.sqrt(a ** 2 + b ** 2 - (2 * a * b * np.cos(theta)))
            mask = np.logical_or.reduce([
                c < self.c_min,
                c > self.c_max,
                ])
            sample_mesh = (a, b, c)
            values_slice = self.nds(np.stack(sample_mesh, axis=-1))
            values_slice[mask] = 0.0
            values.append(values_slice)
        self.values = values

        pos = np.stack(values)
        pos = pos[pos != 0]
        neg = -pos.copy()
        neg[neg < 0] = 0
        pos[pos < 0] = 0
        self.vscale = np.min([np.std(neg), np.std(pos)])
    def plot_slices(self, axes=None, half=False, **kwargs):
        if self.values is None:
            raise ValueError("Values must be generated with sample_uniformly.")

        default_kwargs = dict(vmin=-0.1,
                              vmax=0.1,
                              cmap="RdBu_r")
        default_kwargs.update(kwargs)

        if axes is None:
            axes = [plt.subplots(figsize=(3.5, 3.5))[1]
                    for _ in range(self.n_cuts)]
        for i in range(self.n_cuts):
            ax = axes[i]

            theta = self.thetas[i]
            ax.set_title(fr"$\theta$ = {theta:.0f}°")
            grid = self.values[i]

            if half == "upper" or half == True:
                grid = np.triu(grid)
            elif half == "lower":
                grid = np.tril(grid)

            x_plot = self.mesh[0][0, :]
            y_plot = self.mesh[1][:, 0]
            ax.imshow(grid, extent=(x_plot[0], x_plot[-1],
                                    y_plot[0], y_plot[-1]),
                      origin="lower",
                      **default_kwargs,
                      )
            ax.set_xlabel("$r_{ij}$")
            ax.set_ylabel("$r_{ik}$")
        return axes




def find_clusters(array):
    clustered = np.empty_like(array)
    unique_vals = np.unique(array)
    cluster_count = 0
    for val in unique_vals:
        labelling, label_count = ndimage.label(array == val)
        for k in range(1, label_count + 1):
            clustered[labelling == k] = cluster_count
            cluster_count += 1
    return clustered, cluster_count


def find_centroids(x_grid, y_grid, z_grid, cutoff=0.001, n_max=2):
    grid_raw = z_grid.copy()
    z_grid[z_grid < cutoff] = 0
    z_grid[z_grid >= cutoff] = 1

    clustered, cluster_count = find_clusters(z_grid)
    c_selection = []
    c_scores = []
    for j in range(1, cluster_count):
        mask = (clustered == j)
        if np.sum(mask) == 0:
            continue
        val_cluster = np.abs(grid_raw[mask])
        c_selection.append(j)
        c_scores.append(np.max(val_cluster))
    c_selection = np.array(c_selection)
    c_scores = np.array(c_scores)
    c_sort = np.argsort(c_scores)[::-1]
    c_selection = c_selection[c_sort[:n_max]]

    centroids = []
    for j in c_selection:
        mask = (clustered == j)
        xx = x_grid[mask]
        yy = y_grid[mask]
        zz = grid_raw[mask]
        idx_min = np.argmax(np.abs(zz))

        x_centroid = xx[idx_min]
        y_centroid = yy[idx_min]
        z_centroid = zz[idx_min]

        centroids.append([x_centroid, y_centroid, z_centroid])
    return np.array(centroids)


def make_triangle(a, b, c=None, theta=None, angle=None, center=None, arc=None):
    if theta is None and c is not None:
        theta = np.arccos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b))
    elif theta is not None and c is None:
        theta = np.deg2rad(theta)
    elif theta is None and c is None:
        raise ValueError

    if arc is None:
        arc = np.min([a, b]) * 0.5

    r = [[0.0, 0.0],
         [a, 0.0],
         [np.cos(theta) * b, np.sin(theta) * b]]
    r = np.array(r)

    theta_arc = np.linspace(0.0, theta)
    arc = np.column_stack([np.cos(theta_arc) * arc,
                           np.sin(theta_arc) * arc])

    if center is not None:
        rot = [[np.cos(theta / 2), -np.sin(theta / 2)],
               [np.sin(theta / 2), np.cos(theta / 2)]]
        r = np.dot(r, rot)
        arc = np.dot(arc, rot)

    if angle is not None:
        rot = [[np.cos(angle), -np.sin(angle)],
               [np.sin(angle), np.cos(angle)]]
        r = np.dot(r, rot)
        arc = np.dot(arc, rot)
    return r, arc


def draw_triangle(r,
                  arc,
                  r_max=3.5,
                  ax=None,
                  scatters=None,
                  lines=None,
                  arcs=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(1.5, 1.5))
        ax.axis("off")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_xlim(-r_max * 1.1, r_max * 1.1)
        ax.set_ylim(-r_max * 1.1, r_max * 1.1)

    scatter_config = dict(s=50, c="gray", linewidth=1, edgecolor="k")
    line_config = dict(color="k", linewidth=1)
    arc_config = dict(color="k", linewidth=1)

    if scatters is not None:
        scatter_config.update(scatters)
    if lines is not None:
        line_config.update(lines)
    if arcs is not None:
        arc_config.update(arcs)
    scatters = scatter_config
    lines = line_config
    arcs = arc_config

    atm = ax.scatter(r[:, 0], r[:, 1], **scatters, zorder=101)
    atm.set_clip_on(False)

    ax.plot([r[0, 0], r[1, 0]], [r[0, 1], r[1, 1]], **lines, zorder=100)
    ax.plot([r[0, 0], r[2, 0]], [r[0, 1], r[2, 1]], **lines, zorder=100)
    ax.plot([r[0, 0], r[2, 0]], [r[0, 1], r[2, 1]], **lines, zorder=100)

    ax.plot(arc[:, 0], arc[:, 1], **arcs)
    return ax


def plot_connections(theta,
                     v,
                     cx,
                     cy,
                     ax,
                     ax_square,
                     xyB,
                     r_max,
                     vscale=0.500,
                     sign="+",
                     ):
    cval = np.clip(v, -vscale, vscale)
    cval = cval / (vscale * 2) + 0.5
    cval = cm.RdBu_r(cval)
    ax_square.scatter([cx], [cy], color=[cval], edgecolor="k")
    triangle, arc = make_triangle(cx, cy, theta=theta)
    draw_triangle(triangle,
                  arc,
                  ax=ax,
                  scatters=dict(s=10, c=[cval]))
    ax.text(0.5,
            0.25,
            f"{sign}{v:.2f}",
            ha="center",
            va="center",
            fontsize=8,
            transform=ax.transAxes)
    ax.axis("equal")
    ax.set_xlim(-r_max, r_max)
    ax.set_ylim(-r_max, r_max)
    xyA = (cx, cy)
    coordsA = ax_square.transData
    coordsB = ax.transData
    con = ConnectionPatch(xyA,
                          xyB,
                          coordsA,
                          coordsB,
                          arrowstyle="->",
                          color="lightgray"
                          )
    return con
