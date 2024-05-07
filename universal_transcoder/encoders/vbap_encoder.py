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

import numpy as np
from scipy.spatial import Delaunay

from universal_transcoder.auxiliars.my_coordinates import MyCoordinates


def vbap_2D_encoder(point: MyCoordinates, layout: MyCoordinates):
    """Function to obtain the gains to encode in multichannel,
    for the speaker layout introduced (2D), using VBAP

    Args:
        point (MyCoordinates): position of the virtual source
        layout (MyCoordinates): positions of speaker layout, ranges: azimut [-180º,180º]
                and elevation [-90º,90º] of each speaker

    Returns:
        output (numpy array): gains for each speaker, sorted in the same order
                as was introduced in the input layout
    """

    theta = point.sph_deg()[0][0]
    layout_sph = layout.get_sph(convention="top_elev", unit="deg")[:, 0]  # only thetas
    layout_sph_sorted = np.sort(layout_sph)  # Sort in ascending azimut order
    sort_positions = np.argsort(layout_sph)  # Save original positions

    # Pair selection
    pair = None
    # Check all pair of speakers in the array
    for i in range(len(layout_sph_sorted) - 1):
        if theta >= layout_sph_sorted[i] and theta <= layout_sph_sorted[i + 1]:
            pair = (
                sort_positions[i],
                sort_positions[i + 1],
            )  # Assign original positions
            break

    # If no pair was selected, then first and last speaker form a pair
    if pair is None:
        pair = (sort_positions[0], sort_positions[-1])  # Assign original positions

    # Polar to cartesian coordinates for input position 'p' (theta)
    p = point.cart()[0][0:2]
    if np.allclose(p, [0, 0], atol=0.0001):
        p = [-1, 0]  # to the back

    # Cartesian coordinates for speakers positions (theta)
    layout_cart = layout.cart()
    L_12 = np.array([layout_cart[pair[0], 0:2], layout_cart[pair[1], 0:2]])

    # Gains vector for the pair
    g = np.dot(p, np.linalg.inv(L_12))

    # Gains normalization
    g_out = abs(g / np.sum(abs(g)))

    # Prepare output array
    output = np.zeros(layout_cart.shape[0])
    output[pair[0]] = g_out[0]
    output[pair[1]] = g_out[1]

    return output


def vbap_3D_encoder(point: MyCoordinates, layout: MyCoordinates):
    """Function to obtain the gains to encode in multichannel,
    for the speaker layout introduced (3D), using VBAP

    Args:
        point (MyCoordinates): position of the virtual source
        layout (MyCoordinates): positions of speaker layout, ranges: azimut [-180º,180º]
                and elevation [-90º,90º] of each speaker

    Returns:
        output (numpy array): gains for each speaker, sorted in the same order
                as was introduced in the input layout
    """

    # Polar to cartesian coordinates for input position 'p' (theta,phi)
    p = point.cart()[0]

    # Triangulation of surface
    layout_cart = layout.cart()
    triangles = np.sort(Delaunay(layout_cart).convex_hull)

    # Gains calculation
    # Set starting value to a very low value
    g_min = -2

    for T in range(len(triangles)):
        # Current triangule
        now = triangles[T]

        # If speakers of triangule are co-linear, ignore them
        if (
            layout_cart[now[0]][2] != 0
            or layout_cart[now[1]][2] != 0
            or layout_cart[now[2]][2] != 0
        ):
            L_123 = np.array(
                [layout_cart[now[0]], layout_cart[now[1]], layout_cart[now[2]]]
            )

            # Gains vector for the triangle
            g = np.dot(p, np.linalg.inv(L_123))

            # Check if min 'g' is the highest g_min for the moment
            g_min_check = np.min(g)
            if g_min_check > g_min:
                # If it is, g_min g_out vector and speakers of triangule are saved
                g_min = g_min_check
                g_out = g
                speakers = now

    # Gains normalization
    g_out = g_out / np.sum(g_out)

    # Prepare output array
    output = np.zeros(layout_cart.shape[0])
    output[speakers[0]] = g_out[0]
    output[speakers[1]] = g_out[1]
    output[speakers[2]] = g_out[2]

    return output
