"""
Copyright (c) 2024 Dolby Laboratories, Amaia Sagasti
Copyright (c) 2023 Dolby Laboratories

Redistribution and use in source and binary forms, with or without modification, are permitted 
provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions 
and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or 
promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED 
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A 
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR 
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED 
TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING 
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
POSSIBILITY OF SUCH DAMAGE.
"""

import math
import ssl

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter

from universal_transcoder.auxiliars.my_coordinates import MyCoordinates
from universal_transcoder.auxiliars.typing import Array
from universal_transcoder.auxiliars.typing import ArrayLike
from universal_transcoder.calculations.energy_intensity import (
    angular_error,
    width_angle,
)
from universal_transcoder.plots_and_logs.common_plots_functions import save_plot


def plot_ei_2D(
    energy: Array,
    radial_i: Array,
    transverse_i: Array,
    cloud_points: MyCoordinates,
    save_results: bool,
    results_file_name=False,
):
    """
    Function to plot the energy and intensity when
    decoding from an input format to an output layout. 2D plots.

    Args:
        energy (Array): contains the energy values for each virtual source (1xL)
        radial_i (Array): contains the radial intensity values for each virtual
                source (1xL)
        transverse_i (array): contains the transversal intensity values for each virtual
                source (1xL)
        cloud(MyCoordinates): set of points sampling the sphere (L)
        save_results (bool): Flag to save plots
        results_file_name(str): Path where to save the plots
    """
    # Preparation
    azimuth = cloud_points.sph_rad()[:, 0]
    elevation = cloud_points.sph_rad()[:, 1]
    mask_horizon = (elevation < 0.01) * (elevation > -0.01)

    # Calculations
    # Energy dB
    energy_db = 10 * np.log10(energy)
    # Angular Error
    ang_err = angular_error(radial_i, transverse_i)
    ang_err_rad = ang_err * math.pi / 180
    ang_err_rad = ang_err_rad
    # Width Angle
    width_deg = width_angle(radial_i) / 3
    width_rad = width_deg * math.pi / 180
    width_rad = width_rad
    # Maxima
    maxima = np.max(np.array([np.max(energy), np.max(radial_i), np.max(transverse_i)]))
    if maxima < 0.9:
        lim = 1.0
    elif np.isclose(maxima, 1.0, rtol=1e-09, atol=1e-09):
        lim = 1.1
    else:
        lim = maxima + 0.1

    # mask point at the horizon
    azimuth = azimuth[mask_horizon]
    energy = energy[mask_horizon]
    energy_db = energy_db[mask_horizon]
    radial_i = radial_i[mask_horizon]
    transverse_i = transverse_i[mask_horizon]
    ang_err_rad = ang_err_rad[mask_horizon]
    width_rad = width_rad[mask_horizon]
    ang_err = ang_err[mask_horizon]
    width_deg = width_deg[mask_horizon]

    # Add last to close plot
    azimuth = np.hstack((azimuth, azimuth[0]))
    energy = np.hstack((energy, energy[0]))
    energy_db = np.hstack((energy_db, energy_db[0]))
    radial_i = np.hstack((radial_i, radial_i[0]))
    transverse_i = np.hstack((transverse_i, transverse_i[0]))
    ang_err_rad = np.hstack((ang_err_rad, ang_err_rad[0]))
    width_rad = np.hstack((width_rad, width_rad[0]))
    ang_err = np.hstack((ang_err, ang_err[0]))
    width_deg = np.hstack((width_deg, width_deg[0]))

    # Plots
    # Plot 1 - Energy, Radial, Transverse and Angular Error
    fig = plt.figure(figsize=(17, 9))
    ax = fig.add_subplot(221, projection="polar")
    ax.plot(azimuth, energy, label="Energy")
    ax.plot(azimuth, radial_i, label="Radial Intensity")
    ax.plot(azimuth, transverse_i, label="Transversal Intensity")
    radii = np.ones(ang_err.size) * lim
    cmap = plt.cm.Greens
    normalize = Normalize(vmin=0.0, vmax=math.pi / 4)
    for t, r, w, d in zip(azimuth, radii, ang_err_rad, ang_err):
        color = cmap(normalize(w))
        bar = ax.bar(t, r, width=w, bottom=0.0, edgecolor="none")
        bar[0].set_facecolor(color)

    def deg_formatter(x, pos):
        return f"{np.rad2deg(x):.0f}°"

    cbar = plt.colorbar(
        plt.cm.ScalarMappable(norm=normalize, cmap=cmap),
        ax=ax,
        pad=0.15,
        shrink=0.75,
        format=FuncFormatter(deg_formatter),
    )
    cbar.ax.set_ylabel("Angular Error")
    ax.legend(bbox_to_anchor=(1.7, 1.3))
    ax.set_theta_zero_location("N")
    plt.title("Energy, Intensity and Angular Error")
    plt.ylim(0, lim)

    # Plot 2 - Energy, Radial, Transverse and Width Angle
    ax = fig.add_subplot(223, projection="polar")
    ax.plot(azimuth, energy, label="Energy")
    ax.plot(azimuth, radial_i, label="Radial Intensity")
    ax.plot(azimuth, transverse_i, label="Transversal Intensity")
    radii = np.ones(width_deg.size) * lim
    cmap = plt.cm.Purples
    normalize = Normalize(vmin=0.0, vmax=math.pi / 6)
    for t, r, w, d in zip(azimuth, radii, width_rad, width_deg):
        color = cmap(normalize(w))
        bar = ax.bar(t, r, width=w, bottom=0.0, edgecolor="none")
        bar[0].set_facecolor(color)

    def deg_formatter(x, pos):
        return f"{np.rad2deg(x)*3:.0f}°"

    cbar = plt.colorbar(
        plt.cm.ScalarMappable(norm=normalize, cmap=cmap),
        ax=ax,
        pad=0.15,
        shrink=0.75,
        format=FuncFormatter(deg_formatter),
    )
    cbar.ax.set_ylabel("Angular Width")
    ax.set_theta_zero_location("N")
    plt.title("Energy, Intensity and Width")
    plt.ylim(0, lim)

    # Plot 3 - Energy dB
    ax = fig.add_subplot(222, projection="polar")
    ax.plot(azimuth, energy_db)
    ax.legend(bbox_to_anchor=(1.5, 1.1))
    ax.set_theta_zero_location("N")
    plt.title("Energy(dBs)")
    # ax.set_rticks([0,1,2,3])
    plt.ylim(-24, 6)

    # Save plots
    if save_results and (type(results_file_name) == str):
        file_name = "plot_energy_intensity_2D.png"
        save_plot(plt, results_file_name, file_name)


def plot_ei_3D(
    energy: ArrayLike,
    radial_i: ArrayLike,
    transverse_i: ArrayLike,
    cloud_points: MyCoordinates,
    save_results: bool,
    results_file_name=False,
):
    """
    Function to plot the energy and intensity when
    decoding from an input format to an output layout.
    3D projection on 2D plots.

    Args:
        energy (jax.numpy Array): contains the energy values for each virtual source (1xL)
        radial_i (jax.numpy array): contains the radial intensity values for each virtual
                source (1xL)
        transverse_i (jax.numpy array): contains the transversal intensity values for each virtual
                source (1xL)
        cloud(MyCoordinates): set of points sampling the sphere (L)
        save_results (bool): Flag to save plots
        results_file_name(str): Path where to save the plots
    """
    from mpl_toolkits.basemap import Basemap

    # Preparation
    points = cloud_points.sph_deg()

    # Calculations
    # Angular Error
    ang_err = angular_error(radial_i, transverse_i)
    # Width Angle
    width_deg = width_angle(radial_i)

    # Plot
    ssl._create_default_https_context = ssl._create_unverified_context
    x = points[:, 0]
    y = points[:, 1]
    # Plot Energy
    fig = plt.figure(figsize=(17, 9))
    ax = fig.add_subplot(321)
    m = Basemap(projection="robin", lon_0=0, resolution="c")
    x_map, y_map = m(x, y)
    m.drawcoastlines(linewidth=0.5, color="None")
    m.drawparallels(
        np.arange(-90.0, 120.0, 45.0),
        labels=[True, True, False, False],
        labelstyle="+/-",
    )
    m.drawmeridians(
        np.arange(0.0, 360.0, 60.0),
        labels=[False, False, False, True],
        labelstyle="+/-",
    )
    plt.scatter(
        x_map,
        y_map,
        s=60 - 0.6 * (np.abs(y)),  # for every 5 angle
        c=energy,
        cmap="coolwarm",
        alpha=0.8,
        edgecolors="none",
    )  # s=6 - 0.05 * (np.abs(y)), for every angle
    plt.colorbar(label="Energy")
    plt.title("Energy")
    plt.clim(0, 2)

    # Plot Radial Intensity
    ax2 = fig.add_subplot(323)
    m = Basemap(projection="robin", lon_0=0, resolution="c")
    x_map, y_map = m(x, y)
    m.drawcoastlines(linewidth=0.5, color="None")
    m.drawparallels(
        np.arange(-90.0, 120.0, 45.0),
        labels=[True, True, False, False],
        labelstyle="+/-",
    )
    m.drawmeridians(
        np.arange(0.0, 360.0, 60.0),
        labels=[False, False, False, True],
        labelstyle="+/-",
    )
    plt.scatter(
        x_map,
        y_map,
        s=60 - 0.6 * (np.abs(y)),  # for every 5 angle
        c=radial_i,
        cmap="coolwarm",
        alpha=0.8,
        edgecolors="none",
    )  # s=6 - 0.05 * (np.abs(y)), for every angle
    plt.colorbar(label="Radial Intensity")
    plt.title("Radial Intensity")
    plt.clim(0, 1)

    # Plot Transverse Intensity
    ax3 = fig.add_subplot(325)
    m = Basemap(projection="robin", lon_0=0, resolution="c")
    x_map, y_map = m(x, y)
    # Dibujar la costa y los límites políticos
    m.drawcoastlines(linewidth=0.5, color="None")
    m.drawparallels(
        np.arange(-90.0, 120.0, 45.0),
        labels=[True, True, False, False],
        labelstyle="+/-",
    )
    m.drawmeridians(
        np.arange(0.0, 360.0, 60.0),
        labels=[False, False, False, True],
        labelstyle="+/-",
    )
    plt.scatter(
        x_map,
        y_map,
        s=60 - 0.6 * (np.abs(y)),  # for every 5 angle
        c=transverse_i,
        cmap="coolwarm",
        alpha=0.8,
        edgecolors="none",
    )  # s=6 - 0.05 * (np.abs(y)), for every angle
    plt.colorbar(label="Transversal Intensity")
    plt.title("Transverse Intensity")
    plt.clim(0, 1)

    # Plot Angular Error
    ax4 = fig.add_subplot(326)
    m = Basemap(projection="robin", lon_0=0, resolution="c")
    x_map, y_map = m(x, y)
    m.drawcoastlines(linewidth=0.5, color="None")
    m.drawparallels(
        np.arange(-90.0, 120.0, 45.0),
        labels=[True, True, False, False],
        labelstyle="+/-",
    )
    m.drawmeridians(
        np.arange(0.0, 360.0, 60.0),
        labels=[False, False, False, True],
        labelstyle="+/-",
    )
    plt.scatter(
        x_map,
        y_map,
        s=60 - 0.6 * (np.abs(y)),  # for every 5 angle
        c=ang_err,
        cmap="coolwarm",
        alpha=0.8,
        edgecolors="none",
    )  # s=6 - 0.05 * (np.abs(y)), for every angle
    plt.colorbar(label="Angular Error (Degrees)")
    plt.title("Angular Error")
    plt.clim(0, 45)

    # Plot Width Angle
    ax5 = fig.add_subplot(324)
    m = Basemap(projection="robin", lon_0=0, resolution="c")
    x_map, y_map = m(x, y)
    m.drawcoastlines(linewidth=0.5, color="None")
    m.drawparallels(
        np.arange(-90.0, 120.0, 45.0),
        labels=[True, True, False, False],
        labelstyle="+/-",
    )
    m.drawmeridians(
        np.arange(0.0, 360.0, 60.0),
        labels=[False, False, False, True],
        labelstyle="+/-",
    )
    plt.scatter(
        x_map,
        y_map,
        s=60 - 0.6 * (np.abs(y)),  # for every 5 angle
        c=width_deg,
        cmap="coolwarm",
        alpha=0.8,
        edgecolors="none",
    )  # s=6 - 0.05 * (np.abs(y)), for every angle
    plt.colorbar(label="Width Source (Degrees)")
    plt.title("Width Source")
    plt.clim(0, 45)

    # Show
    # plt.show()
    ssl._create_default_https_context = ssl._create_default_https_context

    # Save plots
    if save_results and (type(results_file_name) == str):
        file_name = "plot_energy_intensity_3D.png"
        save_plot(plt, results_file_name, file_name)
