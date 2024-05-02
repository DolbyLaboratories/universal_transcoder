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

import numpy as np

from universal_transcoder.auxiliars.my_coordinates import MyCoordinates
from universal_transcoder.auxiliars.typing import NpArray


def ambisonics_encoder(point: MyCoordinates, n: int) -> NpArray:
    """Function to obtain the channel gains for the corresponding Ambisonics
     order,sorted in ACN format and normalized following SN3D

    Args:
        point (MyCoordinates): position of the virtual source
        n (int): desired ambisonics order

    Returns:
        encoder_matrix (numpy array): gains for each ambisonics channel
    """

    coord = point.sph_rad()
    theta = coord[0][0]

    phi = coord[0][1]

    encoder_matrix = []

    # 0 - order
    if n >= 0:
        encoder_matrix.append(1.0)  # W

    # 1 - order
    if n >= 1:
        encoder_matrix.append(math.sin(theta) * math.cos(phi))  # Y
        encoder_matrix.append(math.sin(phi))  # Z
        encoder_matrix.append(math.cos(theta) * math.cos(phi))  # X

    # 2 - order
    if n >= 2:
        encoder_matrix.append(
            math.sqrt(3) / 2 * math.sin(2 * theta) * math.cos(phi) ** 2
        )  # V
        encoder_matrix.append(
            math.sqrt(3) / 2 * math.sin(theta) * math.sin(2 * phi)
        )  # T
        encoder_matrix.append((3 * (math.sin(phi) ** 2) - 1) / 2)  # R
        encoder_matrix.append(
            math.sqrt(3) / 2 * math.cos(theta) * math.sin(2 * phi)
        )  # S
        encoder_matrix.append(
            math.sqrt(3) / 2 * math.cos(2 * theta) * math.cos(phi) ** 2
        )  # U

    # 3 - order
    if n >= 3:
        encoder_matrix.append(
            math.sqrt(5 / 8) * math.sin(3 * theta) * math.cos(phi) ** 3
        )  # Q
        encoder_matrix.append(
            math.sqrt(15) / 2 * math.sin(2 * theta) * math.sin(phi) * math.cos(phi) ** 2
        )  # O
        encoder_matrix.append(
            math.sqrt(3 / 8)
            * math.sin(theta)
            * math.cos(phi)
            * (5 * math.sin(phi) ** 2 - 1)
        )  # M
        encoder_matrix.append(math.sin(phi) * (5 * math.sin(phi) ** 2 - 3) / 2)  # K
        encoder_matrix.append(
            math.sqrt(3 / 8)
            * math.cos(theta)
            * math.cos(phi)
            * (5 * math.sin(phi) ** 2 - 1)
        )  # L
        encoder_matrix.append(
            math.sqrt(15) / 2 * math.cos(2 * theta) * math.sin(phi) * math.cos(phi) ** 2
        )  # N
        encoder_matrix.append(
            math.sqrt(5 / 8) * math.cos(3 * theta) * math.cos(phi) ** 3
        )  # P

    # 4 - order
    if n >= 4:
        encoder_matrix.append(
            math.sqrt(35.0 / 2.0)
            * 3.0
            / 8.0
            * math.sin(4.0 * theta)
            * math.cos(phi) ** 4
        )  # (4,-4)
        encoder_matrix.append(
            math.sqrt(35.0)
            * 3.0
            / 4.0
            * math.sin(3.0 * theta)
            * math.sin(phi)
            * math.cos(phi) ** 3
        )  # (4,-3)
        encoder_matrix.append(
            math.sqrt(5.0 / 2.0)
            / 4.0
            * (
                -3.0 * math.sin(2.0 * theta) * math.sin(phi) * math.cos(phi) ** 3
                + 21.0 * math.sin(2.0 * theta) * math.sin(phi) ** 2 * math.cos(phi) ** 2
            )
        )  # (4,-2)
        encoder_matrix.append(
            math.sqrt(5.0)
            / 4.0
            * (
                21.0 * math.sin(theta) * math.cos(phi) * math.sin(phi) ** 3
                - 9.0 * math.sin(theta) * math.cos(phi) * math.sin(phi)
            )
        )  # (4,-1)
        encoder_matrix.append(
            3.0 / 64.0 * (20.0 * math.cos(2.0 * phi) - 35.0 * math.cos(4.0 * phi) - 9.0)
        )  # (4,-0)
        encoder_matrix.append(
            math.sqrt(5.0)
            / 4.0
            * (
                21.0 * math.cos(theta) * math.cos(phi) * math.sin(phi) ** 3
                - 9.0 * math.cos(theta) * math.cos(phi) * math.sin(phi)
            )
        )  # (4, 1)
        encoder_matrix.append(
            math.sqrt(5.0 / 2.0)
            / 4.0
            * (
                -3.0 * math.cos(2.0 * theta) * math.sin(phi) * math.cos(phi) ** 3
                + 21.0 * math.cos(2.0 * theta) * math.sin(phi) ** 2 * math.cos(phi) ** 2
            )
        )  # (4, 2)
        encoder_matrix.append(
            math.sqrt(35.0)
            * 3.0
            / 4.0
            * math.cos(3.0 * theta)
            * math.sin(phi)
            * math.cos(phi) ** 3
        )  # (4, 3)
        encoder_matrix.append(
            math.sqrt(35.0 / 2.0)
            * 3.0
            / 8.0
            * math.cos(4.0 * theta)
            * math.cos(phi) ** 4
        )  # (4, 4)

    # 5 - order
    if n >= 5:
        encoder_matrix.append(
            3.0 / 16.0 * math.sqrt(77.0) * math.sin(5.0 * theta) * math.cos(phi) ** 5
        )  # (5,-5)
        encoder_matrix.append(
            3.0
            / 8.0
            * math.sqrt(385.0 / 2.0)
            * math.sin(4.0 * theta)
            * math.sin(phi)
            * math.cos(phi) ** 4
        )  # (5,-4)
        encoder_matrix.append(
            math.sqrt(385.0)
            / 16.0
            * (9.0 * math.sin(3.0 * theta) * math.cos(phi) ** 3 * math.sin(phi) ** 2)
        )  # (5, -3)
        encoder_matrix.append(
            math.sqrt(1155.0 / 2.0)
            / 4.0
            * (
                3.0 * math.sin(2.0 * theta) * math.cos(phi) ** 2 * math.sin(phi) ** 3
                - math.sin(2.0 * theta) * math.cos(phi) ** 2 * math.sin(phi)
            )
        )  # (5,-2)
        encoder_matrix.append(
            math.sqrt(165.0 / 2.0)
            / 8.0
            * (
                math.sin(theta) * math.cos(phi)
                - 7.0 * math.sin(theta) * math.cos(phi) * math.sin(phi) ** 2
                + 21.0 * math.sin(theta) * math.cos(phi) * math.sin(phi) ** 4
            )
        )  # (5,-1)
        encoder_matrix.append(
            math.sqrt(11.0)
            / 8.0
            * (
                15.0 * math.sin(phi)
                - 70.0 * math.sin(phi) ** 3
                + 63.0 * math.sin(phi) ** 5
            )
        )  # (5, 0)
        encoder_matrix.append(
            math.sqrt(165.0 / 2.0)
            / 8.0
            * (
                math.cos(theta) * math.cos(phi)
                - 7.0 * math.cos(theta) * math.cos(phi) * math.sin(phi) ** 2
                + 21.0 * math.cos(theta) * math.cos(phi) * math.sin(phi) ** 4
            )
        )  # (5, 1)
        encoder_matrix.append(
            math.sqrt(1155.0 / 2.0)
            / 4.0
            * (
                3.0 * math.cos(2.0 * theta) * math.cos(phi) ** 2 * math.sin(phi) ** 3
                - math.cos(2.0 * theta) * math.cos(phi) ** 2 * math.sin(phi)
            )
        )  # (5, 2)
        encoder_matrix.append(
            math.sqrt(385.0)
            / 16.0
            * (9.0 * math.cos(3.0 * theta) * math.cos(phi) ** 3 * math.sin(phi) ** 2)
        )  # (5, 3)
        encoder_matrix.append(
            3.0
            / 8.0
            * math.sqrt(385.0 / 2.0)
            * math.cos(4.0 * theta)
            * math.sin(phi)
            * math.cos(phi) ** 4
        )  # (5, 4)
        encoder_matrix.append(
            3.0 / 16.0 * math.sqrt(77.0) * math.cos(5.0 * theta) * math.cos(phi) ** 5
        )  # (5, 5)

    return np.asarray(encoder_matrix)
