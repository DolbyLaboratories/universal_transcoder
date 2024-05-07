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

import ssl
from typing import Union

import matplotlib.pyplot as plt
import numpy as np

from universal_transcoder.auxiliars.my_coordinates import MyCoordinates
from universal_transcoder.auxiliars.typing import ArrayLike, Array
from universal_transcoder.plots_and_logs.common_plots_functions import save_plot


def plot_pv_2D(
    pressure: Array,
    radial_v: Array,
    transverse_v: Array,
    cloud_points: MyCoordinates,
    save_results: bool,
    results_file_name: Union[bool, str] = False,
):
    """
    Function to plot the pressure and velocity when
    decoding from an input format to an output layout. 2D plots.

    Args:
        pressure (jax.numpy Array): contains the real pressure values for each virtual source (1xL)
        radial_v (jax.numpy array): contains the real adial velocity values for each virtual
                source (1xL)
        transverse_v (jax.numpy array): contains the real transversal velocity values for each virtual
                source (1xL)
        cloud(MyCoordinates): set of points sampling the sphere (L)
        save_results (bool): Flag to save plots
        results_file_name(str): Path where to save the plots
    """

    azimuth = cloud_points.sph_rad()[:, 0]
    elevation = cloud_points.sph_rad()[:, 1]
    mask_horizon = np.isclose(np.abs(elevation), np.min(np.abs(elevation))) & (
        elevation >= 0.0
    )

    # Calculations
    v = np.sqrt(radial_v**2 + transverse_v**2)

    # Maxima
    maxima = np.max(
        np.array(
            [
                np.max(pressure),
                np.max(radial_v),
                np.max(transverse_v),
            ]
        )
    )
    if maxima < 1.0:
        lim = 1.0
    elif maxima == 1.0:
        lim = 1.1
    else:
        lim = maxima + 0.1

    # mask point at the horizon
    azimuth = azimuth[mask_horizon]
    pressure = pressure[mask_horizon]
    radial_v = radial_v[mask_horizon]
    transverse_v = transverse_v[mask_horizon]

    # az_order = np.argsort(azimuth)
    # azimuth = azimuth[az_order]
    # elevation = elevation[az_order]

    # Plot
    # XY
    fig1 = plt.figure(figsize=(17, 9))
    ax = fig1.add_subplot(121)
    ax.plot(azimuth, pressure, label="Pressure")
    ax.plot(azimuth, radial_v, label="Radial Velocity")
    ax.plot(azimuth, transverse_v, label="Transverse Velocity")

    ax.legend(bbox_to_anchor=(1.5, 1.1))
    plt.title("Pressure and Velocity")
    plt.ylim(-0.01, lim)

    # Polar
    # Add last to close plot
    azimuth_polar = np.hstack((azimuth, azimuth[0]))
    pressure_polar = np.hstack((pressure, pressure[0]))
    radial_v_polar = np.hstack((radial_v, radial_v[0]))
    transverse_v_polar = np.hstack((transverse_v, transverse_v[0]))

    ax = fig1.add_subplot(122, projection="polar")
    ax.set_theta_zero_location("N")
    ax.plot(azimuth_polar, pressure_polar, label="Pressure")
    ax.plot(azimuth_polar, radial_v_polar, label="Radial Velocity")
    ax.plot(azimuth_polar, transverse_v_polar, label="Transverse Velocity")

    # plt.show()

    # Save plots
    if save_results and (type(results_file_name) == str):
        file_name = "plot_pressure_velocity_2D.png"
        save_plot(plt, results_file_name, file_name)

    # plot in dB scale
    # Plot
    # XY
    fig1 = plt.figure(figsize=(17, 9))
    ax = fig1.add_subplot(121)
    ax.plot(azimuth, 20 * np.log10(pressure), label="Pressure")
    # ax.plot(azimuth, radial_v, label="Radial Velocity")
    # ax.plot(azimuth, transverse_v, label="Transverse Velocity")

    ax.legend(bbox_to_anchor=(1.5, 1.1))
    plt.title("Pressure and Velocity")
    plt.ylim(-24, +6)

    # Polar
    # Add last to close plot
    azimuth_polar = np.hstack((azimuth, azimuth[0]))
    pressure_polar = np.hstack((pressure, pressure[0]))
    radial_v_polar = np.hstack((radial_v, radial_v[0]))
    transverse_v_polar = np.hstack((transverse_v, transverse_v[0]))

    ax = fig1.add_subplot(122, projection="polar")
    ax.set_theta_zero_location("N")
    ax.plot(azimuth_polar, 20 * np.log10(pressure_polar), label="Pressure")
    # ax.plot(azimuth_polar, radial_v_polar, label="Radial Velocity")
    # ax.plot(azimuth_polar, transverse_v_polar, label="Transverse Velocity")
    plt.ylim(-24, +6)

    # Save plots
    if save_results and (type(results_file_name) == str):
        file_name = "plot_pressure_velocity_2D_dB.png"
        save_plot(plt, results_file_name, file_name)


def plot_pv_3D(
    pressure: ArrayLike,
    radial_v: ArrayLike,
    transverse_v: ArrayLike,
    cloud_points: MyCoordinates,
    save_results: bool,
    results_file_name: Union[bool, str] = False,
):
    """
    Function to plot the pressure and velocity when
    decoding from an input format to an output layout.
    3D projection on 2D plots.

    Args:
        pressure (jax.numpy Array): contains the real pressure values for each virtual source (1xL)
        radial_v (jax.numpy array): contains the real adial velocity values for each virtual
                source (1xL)
        transverse_v (jax.numpy array): contains the real transversal velocity values for each virtual
                source (1xL)
        cloud(MyCoordinates): set of points sampling the sphere (L)
        save_results (bool): Flag to save plots
        results_file_name(str): Path where to save the plots
    """
    from mpl_toolkits.basemap import Basemap

    # Preparation
    points = cloud_points.sph_deg()

    # Calculations

    # Prepare plot
    ssl._create_default_https_context = ssl._create_unverified_context
    x = points[:, 0]
    y = points[:, 1]

    # Plot Pressure
    fig = plt.figure(figsize=(17, 9))
    ax = fig.add_subplot(311)
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
        c=pressure,
        cmap="coolwarm",
        alpha=0.8,
        edgecolors="none",
    )  # s=6 - 0.05 * (np.abs(y)), for every angle
    plt.colorbar(label="Pressure")
    plt.title("Pressure")
    plt.clim(0, 2)

    # Plot Radial Velocity
    ax = fig.add_subplot(312)
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
        c=radial_v,
        cmap="coolwarm",
        alpha=0.8,
        edgecolors="none",
    )  # s=6 - 0.05 * (np.abs(y)), for every angle
    plt.colorbar(label="Radial Velocity")
    plt.title("Radial Velocity")
    plt.clim(0, 1)

    # Plot Transverse Velocity
    ax = fig.add_subplot(313)
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
        c=transverse_v,
        cmap="coolwarm",
        alpha=0.8,
        edgecolors="none",
    )  # s=6 - 0.05 * (np.abs(y)), for every angle
    plt.colorbar(label="Transversal Velocity")
    plt.title("Transverse Velocity")
    plt.clim(0, 1)

    # Show
    # plt.show()
    ssl._create_default_https_context = ssl._create_default_https_context

    # Save plots
    if save_results and (type(results_file_name) == str):
        file_name = "plot_pressure_velocity_3D.png"
        save_plot(plt, results_file_name, file_name)
