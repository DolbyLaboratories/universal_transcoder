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

import os
from math import sqrt

import jax.numpy as jnp
import numpy as np
from scipy import linalg

from universal_transcoder.auxiliars.my_coordinates import MyCoordinates
from universal_transcoder.encoders.ambisonics_encoder import ambisonics_encoder
from universal_transcoder.encoders.vbap_encoder import vbap_2D_encoder
from universal_transcoder.encoders.vbap_encoder import vbap_3D_encoder

os.environ["JAX_ENABLE_X64"] = "1"


# VBAP
def get_vbap_decoder_matrix(layout_input: MyCoordinates, layout_output: MyCoordinates):
    """Function to obtain the decoding matrix, based on VBAP, neccessary to transform
    from a multichannel layout to another

    Args:
        layout_input (pyfar.Coordinates): positions of input speaker layout
                azimut and elevation of each speaker (M speakers)
        layout_output (pyfar.Coordinates): positions of output speaker layout:
                azimut and elevation of each speaker (N speakers)

    Returns:
        decoder (jax.numpy array): decoding matrix of size NxM
    """
    layout_input_sph = layout_input.sph_deg()
    layout_output_sph = layout_output.sph_deg()
    M = layout_input_sph.shape[0]
    N = layout_output_sph.shape[0]
    decoder = np.zeros((N, M), dtype=float)

    # Identify if output layout is 2D or 3D

    # 2D
    if jnp.count_nonzero(layout_output_sph, axis=0)[1] == 0:
        for m in range(M):
            input_speaker = layout_input_sph[m]
            virtual = MyCoordinates.point(input_speaker[0], input_speaker[1])
            decoder[:, m] = vbap_2D_encoder(virtual, layout_output)

    # 3D
    else:
        for m in range(M):
            input_speaker = layout_input_sph[m]
            virtual = MyCoordinates.point(input_speaker[0], input_speaker[1])
            decoder[:, m] = vbap_3D_encoder(virtual, layout_output)

    return jnp.array(decoder)


# Ambisonics to speakers
def get_ambisonics_decoder_matrix(order: int, layout_output: MyCoordinates, type: str):
    """Function to obtain the decoding matrix to decode ambisonics to a given
    speaker layout

    Args:
        order (int): ambisonics order from which to decode
        layout_output (pyfar.Coordinates): positions of output speaker layout:
                azimut and elevation of each speaker (N speakers)

        type: type of decoding matrix, pseudo-inverse or normalized

    Returns:
        decoder (jax.numpy array): decoding matrix of size NxnºambisonicsCh
    """

    M = (order + 1) ** 2
    layout_output_sph = layout_output.sph_deg()
    N = layout_output_sph.shape[0]
    decoder = np.zeros((N, M), dtype=float)
    C = np.zeros((N, M), dtype=float)

    for n in range(N):
        spk = layout_output_sph[n]
        virtual = MyCoordinates.point(spk[0], spk[1])
        ambisonics_enc = ambisonics_encoder(virtual, order)
        C[n, :] = ambisonics_enc

    if type == "norm":
        gLF = []

        # 0 - order
        if order >= 0:
            gLF.append(1.0)  # W

        # 1 - order
        if order >= 1:
            gLF.append(1.0 / sqrt(3.0))  # Y
            gLF.append(1.0 / sqrt(3.0))  # Z
            gLF.append(1.0 / sqrt(3.0))  # X

        # 2 - order
        if order >= 2:
            gLF.append(1.0 / sqrt(5.0))  # V
            gLF.append(1.0 / sqrt(5.0))  # T
            gLF.append(1.0 / sqrt(5.0))  # R
            gLF.append(1.0 / sqrt(5.0))  # S
            gLF.append(1.0 / sqrt(5.0))  # U

        # 3 - order
        if order >= 3:
            gLF.append(1.0 / sqrt(7.0))  # Q
            gLF.append(1.0 / sqrt(7.0))  # O
            gLF.append(1.0 / sqrt(7.0))  # M
            gLF.append(1.0 / sqrt(7.0))  # K
            gLF.append(1.0 / sqrt(7.0))  # L
            gLF.append(1.0 / sqrt(7.0))  # N
            gLF.append(1.0 / sqrt(7.0))  # P

        # 4 - order
        if order >= 4:
            gLF.append(1.0 / sqrt(9.0))  #
            gLF.append(1.0 / sqrt(9.0))  #
            gLF.append(1.0 / sqrt(9.0))  #
            gLF.append(1.0 / sqrt(9.0))  #
            gLF.append(1.0 / sqrt(9.0))  #
            gLF.append(1.0 / sqrt(9.0))  #
            gLF.append(1.0 / sqrt(9.0))  #
            gLF.append(1.0 / sqrt(9.0))  #
            gLF.append(1.0 / sqrt(9.0))  #

        # 5 - order
        if order >= 5:
            gLF.append(1.0 / sqrt(11.0))  #
            gLF.append(1.0 / sqrt(11.0))  #
            gLF.append(1.0 / sqrt(11.0))  #
            gLF.append(1.0 / sqrt(11.0))  #
            gLF.append(1.0 / sqrt(11.0))  #
            gLF.append(1.0 / sqrt(11.0))  #
            gLF.append(1.0 / sqrt(11.0))  #
            gLF.append(1.0 / sqrt(11.0))  #
            gLF.append(1.0 / sqrt(11.0))  #
            gLF.append(1.0 / sqrt(11.0))  #
            gLF.append(1.0 / sqrt(11.0))  #

        gLF = np.array(gLF)
        factor = np.tile(gLF, (N, 1))
        decoder = (1.0 / N) * factor * C
        ret = jnp.array(decoder)

    elif type == "pseudo":
        ret = jnp.array(linalg.pinv(C).T)

    return ret
