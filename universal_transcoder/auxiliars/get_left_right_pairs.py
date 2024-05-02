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

from universal_transcoder.auxiliars.my_coordinates import MyCoordinates


def angular_distance(theta1, e1, theta2, e2):
    """Function to calculate the angular distance between two points.
    Haversine formula (see Wikipedia)

    Args:
        theta1 (float): Radians. Latitude of point 1
        e1 (float): Radians. Longitude of point 1
        theta2 (float): Radians. Latitude of point 2
        e2 (float): Radians. Longitude of point 2

    Returns:
        dist (float): Angular distance
    """
    # Implement haversine formula (see Wikipedia)
    dist = 2.0 * np.arcsin(
        np.sqrt(
            (np.sin((e1 - e2) / 2.0)) ** 2
            + np.cos(e1) * np.cos(e2) * (np.sin((theta1 - theta2) / 2.0)) ** 2
        )
    )
    return dist


def get_left_right_pairs(points: MyCoordinates, tol: float = 2.0):
    """Function to find the symmetric pairs in a layout, a set of positions,
    with a tolerance (default 2 degrees)

    Args:
        layout (MyCoordinates): positions given as pyfar.Coordinates
        tol (float): Degrees of tolerance allowed to find pairs

    Returns:
        paired (Array): array which positions corresponds to those of the
        input layout, and in which points belonging to the same pair are "tagged"
        with the same number
    """
    points_coord = points.sph_rad()
    thetas = points_coord[:, 0]
    phis = points_coord[:, 1]
    tol = tol * np.pi / 180
    pair_id = 1
    pairs = np.zeros(len(thetas))
    for i, theta in enumerate(thetas):
        if abs(pairs[i]) >= 1:
            continue
        else:
            non_zero_pos = np.nonzero(thetas[i + 1 : len(thetas)])[0] + (1 + i)
            for j in non_zero_pos:
                if angular_distance(theta, phis[i], -thetas[j], phis[j]) <= tol:
                    # pair found
                    pairs[i] = pair_id
                    pairs[j] = -pair_id
                    pair_id = pair_id + 1
                    break
    return pairs
