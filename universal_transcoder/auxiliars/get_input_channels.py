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

import jax.numpy as jnp
import numpy as np

from universal_transcoder.auxiliars.my_coordinates import MyCoordinates
from universal_transcoder.auxiliars.typing import NpArray
from universal_transcoder.encoders.ambisonics_encoder import ambisonics_encoder
from universal_transcoder.encoders.vbap_encoder import vbap_2D_encoder, vbap_3D_encoder


def get_input_channels_vbap(
    cloud_points: MyCoordinates,
    input_layout: MyCoordinates,
    normalize: bool = True,
    vbip: bool = False,
) -> np.ndarray:
    """Function to obtain the channel gains that encode in a multichannel layout for a cloud of points, where
    each row corresponds to the coding gains for one point, using VBAP

    Args:
        cloud_points (MyCoordinates): positions of the virtual sources pyfar.Coordinates
                (L points)
        input_layout (MyCoordinates): speaker positions given as pyfar.Coordinates
                (M speakers)
        normalize (bool): If activated, gains are normalized so that sum of all squares equals 1
        vbip (bool): Use vbip instead of vbap

    Returns:
        input_channels (numpy Array): LxM matrix of channel gains for input layout
    """
    input_layout_sph = input_layout.sph_deg()
    cloud_points_sph = cloud_points.sph_deg()
    input_channels = np.zeros([cloud_points_sph.shape[0], input_layout_sph.shape[0]])

    if jnp.count_nonzero(input_layout_sph, axis=0)[1] == 0:
        encoder = vbap_2D_encoder
    else:
        encoder = vbap_3D_encoder

    for i in range(cloud_points_sph.shape[0]):
        virtual = MyCoordinates.point(
            cloud_points_sph[i, 0], cloud_points_sph[i, 1], cloud_points_sph[i, 2]
        )
        input_channels[i, :] = encoder(virtual, input_layout)
        if vbip:
            input_channels[i, :] = np.sqrt(input_channels[i, :])

    if normalize:
        sum2 = np.sqrt(np.sum(input_channels**2, axis=1))
        input_channels = input_channels / sum2[:, np.newaxis]

    return input_channels


def get_input_channels_ambisonics(cloud_points: MyCoordinates, order: int) -> NpArray:
    """Function to obtain the channel gains for a cloud of points that encode in
    a given ambisonics order, where each row corresponds to the coding gains
    for one point in the given Ambi-order

    Args:
        cloud_points (MyCoordinates): positions of the virtual sources pyfar.Coordinates
                (L points)
        order (int): ambisonics order ((order+1)**2 = M channels)

    Returns:
        input_channels (numpy Array): LxM matrix of channel gains for input layout
    """
    cloud_points_sph = cloud_points.sph_deg()
    input_channels = np.zeros([cloud_points_sph.shape[0], (order + 1) ** 2])

    for i in range(cloud_points_sph.shape[0]):
        virtual = MyCoordinates.point(
            cloud_points_sph[i, 0], cloud_points_sph[i, 1], cloud_points_sph[i, 2]
        )
        input_channels[i, :] = ambisonics_encoder(virtual, order)

    return input_channels


def get_input_channels_micro_cardioid(
    cloud: MyCoordinates,
    mic_orientation: NpArray,
    mic_position: MyCoordinates,
    f: float,
) -> NpArray:
    """Function to obtain the gains for a cloud of points that encode in cardioid-microphone, where
    each row corresponds to the coding channels for one point

    Args:
        cloud (MyCoordinates): positions of the virtual sources pyfar.Coordinates
                (L points)
        mic_orientation (numpy Array): array specifying orientation of the microphones
                both in azimut and elevation
        mic_position (MyCoordinates) position of microphones
                pyfar.Coordinates (M microphones)
        f (float): frequency of input sound waves in Hz

    Returns:
        input_channels (numpy Array): LxM matrix of channel gains
                for input layout
    """
    cloud_points_sph = cloud.sph_rad()
    mic_orientation_sph = mic_orientation * np.pi / 180
    mic_position_sph = mic_position.sph_rad()
    input_channels = np.zeros(
        [cloud_points_sph.shape[0], mic_orientation_sph.shape[0]], dtype=np.complex64
    )

    for i in range(cloud_points_sph.shape[0]):
        for j in range(mic_orientation_sph.shape[0]):
            gain = (1 + np.cos(mic_orientation_sph[j, 0] - cloud_points_sph[i, 0])) / 2
            deltaR = mic_position_sph[j, 2] - mic_position_sph[j, 2] * np.cos(
                mic_position_sph[j, 0] - cloud_points_sph[i, 0]
            )
            deltaT = deltaR / 340
            exponent = 2 * np.pi * f * deltaT
            input_channels[i, j] = gain * np.exp(-1j * exponent)

    return input_channels


def get_input_channels_micro_omni(
    cloud: MyCoordinates,
    mic_orientation: NpArray,
    mic_position: MyCoordinates,
    f: float,
):
    """Function to obtain the gains for a cloud of points that encode in  omni-microphone , where
    each row corresponds to the coded channels for one point

    Args:
        cloud (MyCoordinates): positions of the virtual sources pyfar.Coordinates
                (L points)
        mic_orientation (np Array): array specifying orientation of the microphones
                both in azimut and elevation
        mic_position (MyCoordinates) position of microphones
                pyfar.Coordinates (M microphones)
        f (float): frequency of input sound waves in Hz

    Returns:
        input_channels (numpy Array): LxM matrix of channel gains
                for input layout
    """
    cloud_points_sph = cloud.sph_rad()
    mic_orientation_sph = mic_orientation * np.pi / 180
    mic_position_sph = mic_position.sph_rad()
    input_channels = np.zeros(
        [cloud_points_sph.shape[0], mic_orientation_sph.shape[0]], dtype=np.complex64
    )

    for i in range(cloud_points_sph.shape[0]):
        for j in range(mic_orientation_sph.shape[0]):
            gain = 1
            deltaR = mic_position_sph[j, 2] - mic_position_sph[j, 2] * np.cos(
                mic_position_sph[j, 0] - cloud_points_sph[i, 0]
            )
            deltaT = deltaR / 340
            exponent = 2 * np.pi * f * deltaT
            input_channels[i, j] = gain * np.exp(-1j * exponent)

    return input_channels
