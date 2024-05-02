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

from universal_transcoder.auxiliars.my_coordinates import MyCoordinates
from universal_transcoder.auxiliars.typing import ArrayLike
from universal_transcoder.auxiliars.typing import JaxArray


def pressure_calculation(speaker_signals: ArrayLike) -> JaxArray:
    """Function to obtain the real and imaginary pressure of a virtual source in an output
    layout out of the coded channel gains

    Args:
        speaker_signals (Array): speaker signals (P speakers) resulting from decoding
                the input set of encoded L directions (LxP size)

    Returns:
        pressure (jax Array): contains the real pressure values for each virtual source (1xL)
    """

    # Calculate pressure
    pressure = jnp.array(jnp.sum(speaker_signals, axis=1), dtype=jnp.float32)

    return pressure


def velocity_calculation(
    speaker_signals: ArrayLike,
    output_layout: MyCoordinates,
) -> JaxArray:
    """Function to obtain the real and imaginary velocity of a virtual source in an
    output layout out of the coded channel gains

    Args:
        speaker_signals (Array): speaker signals (P speakers) resulting from decoding
                the input set of encoded L directions (LxP size)
        output_layout (MyCoordinates): positions of output speaker layout: (P speakers)

    Returns:
        velocity (jax.numpy array): real velocity vector for each virtual source,
                cartesian coordinates (Lx3)
    """

    # Pressure calculation
    pressure = pressure_calculation(speaker_signals)

    # Cardinal coordinates - unitary vectors of output speakers - Ui
    U = output_layout.cart()

    # Velocity
    aux = jnp.dot(speaker_signals, U)
    velocity = jnp.array((1 / (pressure + jnp.finfo(float).eps)).reshape(-1, 1) * aux)

    return velocity


def radial_V_calculation(
    cloud_points: MyCoordinates,
    speaker_signals: ArrayLike,
    output_layout: MyCoordinates,
) -> JaxArray:
    """Function to obtain the real and imag radial velocity of each virtual source of a cloud of
    points in an output layout out from the coded channel gains

    Args:
        cloud_points (MyCoordinates): position of the virtual sources pyfar.Coordinates (L sources)
        speaker_signals (Array): speaker signals (P speakers) resulting from decoding
                the input set of encoded L directions (LxP size)
        output_layout (MyCoordinates): positions of output speaker layout: (P speakers)

    Returns:
        radial_velocity (jax array): contains the real adial velocity values for each virtual
                source (1xL)
    """
    # Velocity calculation
    velocity = velocity_calculation(speaker_signals, output_layout)

    # Cardinal coordinates direction virtual source - Vj
    V = cloud_points.cart()

    # Scalar product of velocity and V
    radial_velocity = jnp.sum(jnp.multiply(velocity, V), axis=1)

    return radial_velocity


def transversal_V_calculation(
    cloud_points: MyCoordinates,
    speaker_signals: ArrayLike,
    output_layout: MyCoordinates,
) -> JaxArray:
    """Function to obtain the real and imaginary transverse velocity of each virtual source of a cloud of
    points in an output layout out from the coded channel gains

    Args:
        cloud_points (MyCoordinates): position of the virtual sources pyfar.Coordinates (L sources)
        speaker_signals (Array): speaker signals (P speakers) resulting from decoding
                the input set of encoded L directions (LxP size)
        output_layout (MyCoordinates): positions of output speaker layout: (P speakers)

    Returns:
        transversal_velocity (jax array): contains the real transversal velocity values for each virtual
                source (1xL)
    """
    # Velocity calculation
    velocity = velocity_calculation(speaker_signals, output_layout)

    # Cardinal coordinates direction virtual source - Vj
    V = cloud_points.cart()

    # Vectorial product of velocity and V
    transverse_velocity = jnp.array(
        jnp.linalg.norm(jnp.cross(velocity, V, axis=1), axis=1)
    )

    return transverse_velocity
