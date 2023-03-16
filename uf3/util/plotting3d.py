import ndsplines
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go


class ThreeBodyPlotter:
    def __init__(self, model, interaction, theta=True):
        model_dump = model.dump()
        decompressed_coefficients = model_dump["coefficients"][interaction]
        knots_set = model_dump["knots"][interaction]
        self.knots_set = [np.array(k) for k in knots_set]

        self.name = interaction
        self.nds = ndsplines.NDSpline(self.knots_set,
                                      decompressed_coefficients,
                                      3)
        self.knots = knots_set
        self.theta = theta
        self.mesh = None
        self.x_plot = None
        self.y_plot = None
        self.z_plot = None
        self.theta_plot = None
        self.values = None

    def sample_uniformly(self, n_samples):
        if isinstance(n_samples, int):
            n_samples = [n_samples, n_samples, n_samples]
        sample_dims = []
        for k, n in zip(self.knots_set, n_samples):
            vector = np.linspace(k[0], k[-1], n)
            sample_dims.append(vector)
        c_min = np.min(sample_dims[2])
        c_max = np.max(sample_dims[2])

        if self.theta:
            sample_dims[2] = np.linspace(0, np.pi, n_samples[2])

        sample_mesh = np.meshgrid(*sample_dims)

        if self.theta:
            a = sample_mesh[0]
            b = sample_mesh[1]

            print(c_min, c_max)
            theta = sample_mesh[2].copy()
            self.z_plot = theta
            c = np.sqrt(a**2 + b**2 - (2 * a * b * np.cos(theta)))
            mask = np.logical_or(c < c_min, c > c_max)
            sample_mesh = (a, b, c)
        else:
            self.z_plot = sample_mesh[2]

        self.values = self.nds(np.stack(sample_mesh, axis=-1))
        if self.theta:
            self.values[mask] = 0.0
        self.mesh = sample_mesh
        self.x_plot = sample_mesh[0]
        self.y_plot = sample_mesh[1]

    def plot_slices(self, n_slices=5, axes=None, **kwargs):
        if self.values is None:
            raise ValueError("Values must be generated with sample_uniformly.")
        v_slice = np.linspace(np.min(self.z_plot),
                              np.max(self.z_plot),
                              n_slices + 1)

        x_plot = self.x_plot[0, :, 0]
        y_plot = self.y_plot[:, 0, 0]
        z_plot = self.z_plot[0, 0, :]
        values = self.values

        if self.theta:
            ztitle = r"$\theta$ slice"
        else:
            ztitle = "$r_{jk}$ slice"

        default_kwargs = dict(vmin=-0.1,
                              vmax=0.1,
                              cmap="coolwarm")
        default_kwargs.update(kwargs)

        if axes is None:
            axes = [plt.subplots(figsize=(3.5, 3.5))[1]
                    for _ in range(n_slices)]
        for i in range(len(v_slice) - 1):
            ax = axes[i]
            mask = np.logical_and(z_plot >= v_slice[i],
                                  z_plot < v_slice[i + 1])
            z_min = np.rad2deg(v_slice[i])
            z_max = np.rad2deg(v_slice[i + 1])
            ax.set_title(f"{ztitle}: {z_min:.2f}° - {z_max:.2f}°")
            grid = np.mean(values[:, :, mask], axis=2)
            ax.imshow(grid, extent=(x_plot[0], x_plot[-1],
                                    y_plot[0], y_plot[-1]),
                      origin="lower",
                      **default_kwargs,
                      )
            ax.set_xlabel("$r_{ij}$")
            ax.set_ylabel("$r_{ik}$")
        return axes


    def plot_uniform(self,
                     filename=None,
                     title="3B",
                     eye_z=2.3,
                     eye_r=3.25,
                     angle=-30,
                     surface_count=21,
                     opacity=0.1,
                     val_limit=0.1,
                     scale=1,
                     height=400,
                     width=350,
                     aspect=None,
                     margin=None,):
        x_plot = self.x_plot.flatten()
        y_plot = self.y_plot.flatten()
        z_plot = self.z_plot.flatten()
        values = self.values.flatten()

        if self.theta:
            z_plot = np.rad2deg(z_plot)
            ztitle = r"θ<sub>jik</sub>"
            zrange = [0, 180]
            zticks = [30, 60, 90, 120, 150, 180]
        else:
            ztitle = r"r<sub>jk</sub>  [Å]"
            zrange = [0, z_plot[-1]]
            zticks = list(range(0, int(z_plot[-1])))
        x_max = x_plot[-1]
        y_max = y_plot[-1]

        bounds = [np.min(values), np.max(values)]
        if val_limit is None or val_limit < 0.0:
            val_limit = np.min(np.abs(bounds))

        val_scale = np.linspace(-val_limit, val_limit, 5)
        alpha_scale = np.abs(np.linspace(-1, 1, 5) ** 5)
        opacity_scale = np.vstack([val_scale, alpha_scale]).T.tolist()

        fig = go.Figure(data=go.Volume(
            x=x_plot,
            y=y_plot,
            z=z_plot,
            value=values,
            isomin=-val_limit,
            isomax=val_limit,
            opacity=opacity,
            opacityscale=opacity_scale,
            surface_count=surface_count,
            colorscale='RdBu_r',
            hoverinfo='skip',
        ))

        eye_x = eye_r * np.cos(np.deg2rad(angle))
        eye_y = eye_r * np.sin(np.deg2rad(angle))

        camera = dict(
            up=dict(x=0, y=0, z=0.2),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=eye_x, y=eye_y, z=eye_z),
            projection=dict(type="orthographic", ),
        )
        if aspect is None:
            aspect = dict(x=0.7, y=0.7, z=1.4)
        if margin is None:
            margin = dict(r=20, l=10, b=10, t=10)

        fig.update_layout(scene_camera=camera,
                          title={
                              'text': title,
                              'y': 0.95,
                              'x': 0.43,
                              'xanchor': 'center',
                              'yanchor': 'top'},
                          scene_aspectmode="manual",
                          scene_aspectratio=aspect,
                          scene=dict(
                              xaxis=dict(tickvals=list(range(0, int(x_max))),
                                         range=[0, x_max],
                                         backgroundcolor="rgba(0, 0, 0,0)",
                                         gridcolor="lightgray",
                                         showbackground=True),
                              yaxis=dict(tickvals=list(range(0, int(x_max))),
                                         range=[0, y_max],
                                         backgroundcolor="rgba(0, 0, 0,0)",
                                         gridcolor="lightgray",
                                         showbackground=True),
                              zaxis=dict(tickvals=zticks,
                                         backgroundcolor="rgba(0, 0, 0,0)",
                                         gridcolor="lightgray",
                                         range=zrange,
                                         showbackground=True),
                              xaxis_title=r"r<sub>ij</sub>  [Å]",
                              yaxis_title=r"r<sub>ik</sub>  [Å]",
                              zaxis_title=ztitle,
                          ),
                          height=height,
                          width=width,
                          margin=margin,
                          )

        fig.update_traces(colorbar=dict(title=dict(text="[eV]", side="top"),
                                        x=1.0,
                                        xpad=0,
                                        thickness=10, ))

        if filename is not None:
            if filename.endswith("html"):
                fig.write_html(filename)
            else:
                fig.write_image(filename, scale=scale)
        return fig
