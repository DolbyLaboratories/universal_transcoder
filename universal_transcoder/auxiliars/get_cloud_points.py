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

import itertools
import math
import os
from typing import Union, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap

from universal_transcoder.auxiliars.my_coordinates import MyCoordinates
from universal_transcoder.auxiliars.typing import NpArray


def mix_clouds_of_points(
    list_of_cloud_points: Iterable[MyCoordinates],
    list_of_weights: Optional[Iterable[float]] = None,
    discard_lower_hemisphere: bool = True,
) -> Tuple[MyCoordinates, NpArray]:
    """
    Mix several clouds of points, and optionally discard the bottom hemisphere

    Args:
        list_of_cloud_points: List of cloud points to merge.
        list_of_weights: If not None, list of weights for each cloud point.
        discard_lower_hemisphere: If True, discard the lower hemisphere

    Returns:
        Merged cloud of points
        Weight array
    """

    if discard_lower_hemisphere:
        list_of_cloud_points = [
            cp.discard_lower_hemisphere() for cp in list_of_cloud_points
        ]
    npoints = [cp.cart().shape[0] for cp in list_of_cloud_points]
    cloud_points = MyCoordinates.mult_points_cart(
        np.vstack([cp.cart() for cp in list_of_cloud_points])
    )
    if list_of_weights is not None:
        weights = np.asarray(
            list(
                itertools.chain(
                    *[[w / n] * n for n, w in zip(npoints, list_of_weights)]
                )
            )
        )
        weights /= np.mean(weights)
    else:
        weights = np.ones(cloud_points.cart().shape[0])

    return cloud_points, weights


def get_equi_circumference_points(
    num_points: int, plot_show: bool = True
) -> MyCoordinates:
    """Function to obtain the polar coordinates of equidistant
    points of a circumference

    Args:
        num_points (int): number of equidistant points
        plot_show(bool): if True plot of points is shown

    Returns:
        cloud_points (pyfar.Coordinates): points position
    """

    # Obtain angle of separation between 2 points
    angle = 360 / num_points

    # Generate thetas angle
    thetas = angle * np.arange(num_points)

    # Re-order thetas in ascending order and set values from -180 to 180
    # print("thetassss ", thetas)
    thetas = np.where(thetas > 180, thetas - 360, thetas)
    thetas = np.sort(thetas, axis=None)
    thetas = np.where(thetas < 0, thetas + 360, thetas)
    # print("thetassss ", thetas)

    # Generate output(elevation angles=0)
    d = np.ones(thetas.size)
    phis = np.zeros(thetas.size)

    cloud_points = MyCoordinates(
        thetas, phis, d, domain="sph", unit="deg", convention="top_elev"
    )

    # Plot
    if plot_show:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="polar")
        thetas_rad = thetas * math.pi / 180
        ax.scatter(thetas_rad, d)
        plt.title("Equidistant points")
        plt.ylim(0, 1)
        plt.show()

    return cloud_points


def get_equi_t_design_points(
    file_name: Union[str, os.PathLike], plot_show: bool = True
) -> MyCoordinates:
    """Function to obtain the polar and cartesian spherical
    coordinates of equidistant points of a sphere out of a given
    txt from http://neilsloane.com/sphdesigns/

    Args:
        file_name (str): file name and directory, if neccesary
        plot_show(bool): if True plot of points is shown

    Returns:
        cloud_points (MyCoordinates): points position
    """

    content = np.loadtxt(file_name, dtype=float)
    layout_cart = np.zeros([int(content.shape[0] / 3), 3])
    aux = 0
    for i in range(0, content.shape[0], 3):
        layout_cart[aux, 0] = content[i]
        layout_cart[aux, 1] = content[i + 1]
        layout_cart[aux, 2] = content[i + 2]
        aux = aux + 1

    x = layout_cart[:, 0]
    y = layout_cart[:, 1]
    z = layout_cart[:, 2]

    cloud_points = MyCoordinates(x, y, z, domain="cart")
    points = cloud_points.sph_deg()

    # Plot
    if plot_show:
        # Plot 2D
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        fig = plt.figure(figsize=(6, 3))
        m = Basemap(projection="robin", lon_0=0, resolution="c")
        x_map, y_map = m(x, y)
        m.drawcoastlines(linewidth=0.5, color="None")
        # draw parallels and meridians.
        m.drawparallels(
            np.arange(-90.0, 120.0, 45.0),
            labels=[True, True, False, False],
            labelstyle="+/-",
        )
        m.drawmeridians(
            np.arange(0.0, 360.0, 30.0),
            labels=[False, False, False, True],
            labelstyle="+/-",
        )
        plt.scatter(
            x_map,
            y_map,
            s=50,
            c=z,
            cmap="coolwarm",
            alpha=0.8,
            edgecolors="none",
        )
        plt.title("Cloud of points (2D)")

        # Plot 3D
        ax = cloud_points.show()
        plt.title("Cloud of points (3D)")
        plt.show()

    return cloud_points


def get_equi_fibonacci_sphere_points(
    num_points: int, plot_show: bool = True
) -> MyCoordinates:
    """Function to obtain the polar spherical coordinates of
    equidistant points of a sphere

    Args:
        num_points (int): number of equidistant points
        plot_show(bool): if True plot of points is shown

    Returns:
        cloud_points (MyCoordinates): points position
    """

    golden_ang = (3 - np.sqrt(5)) * np.pi  # golden angle

    # Create a list of golden angle increments
    golden_ang = golden_ang * np.arange(num_points)

    # Z ranging [-1,1]
    z = np.linspace(1 / num_points - 1, 1 - 1 / num_points, num_points)

    # radius at each height
    r = np.sqrt(1 - z * z)

    # XY, given the azimuthal and polar angles
    y = r * np.sin(golden_ang)
    x = r * np.cos(golden_ang)

    # Plot
    if plot_show:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(x, y, z)
        plt.show()

    cloud_points = MyCoordinates(x, y, z, domain="cart")

    return cloud_points


def get_all_sphere_points(space: int = 1, plot_show: bool = True) -> MyCoordinates:
    """Function to obtain the coordinates of equidistant points of a
    sphere, with a gap/spacing between them (expressed in degrees) given

    Args:
        space (int): degrees separating each point sampling the sphere
        plot_show(bool): if True plot of points is shown

    Returns:
        cloud_points (pyfar.Coordinates): points position
    """

    points_list = []
    for j in range(-90, 90, space):
        for i in range(-180, 180, space):
            points_list.append([i + 0.5, j + 0.5, 1])
    points = np.array(points_list)

    cloud_points = MyCoordinates.mult_points(points)

    if plot_show:
        # Plot 2D
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        fig = plt.figure(figsize=(6, 3))
        m = Basemap(projection="robin", lon_0=0, resolution="c")
        x_map, y_map = m(x, y)
        m.drawcoastlines(linewidth=0.5, color="None")
        # draw parallels and meridians.
        m.drawparallels(
            np.arange(-90.0, 120.0, 45.0),
            labels=[True, True, False, False],
            labelstyle="+/-",
        )
        m.drawmeridians(
            np.arange(0.0, 360.0, 30.0),
            labels=[False, False, False, True],
            labelstyle="+/-",
        )
        plt.scatter(
            x_map,
            y_map,
            s=50,
            c=z,
            cmap="coolwarm",
            alpha=0.8,
            edgecolors="none",
        )
        plt.title("Cloud of points (2D)")

        # Plot 3D
        ax = cloud_points.show()
        plt.title("Cloud of points (3D)")
        plt.show()

    return cloud_points
