"""
Copyright 2023 Dolby Laboratories

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


def energy_calculation(speaker_signals: jnp):
    """Function to obtain the energy of each virtual source of a cloud of points in an output
    layout out from the coded channel gains

    Args:
        speaker_signals (numpy Array): speaker signals resulting from decoding 
                to input set of encoded L directions (LxN size)

    Returns:
        energy (jax.numpy Array): contains the energy values for each virtual source (1xL)
    """

    # Calculate energy
    energy = jnp.array(jnp.sum(speaker_signals**2, axis=1), dtype=jnp.float32)

    return energy


def intensity_calculation(
    speaker_signals: jnp,
    output_layout: MyCoordinates,
):
    """Function to obtain the intensity of each virtual source of a cloud of points in an output
    layout out from the coded channel gains

    Args:
        speaker_signals (numpy Array): speaker signals resulting from decoding 
                to input set of encoded L directions (LxN size)
        output_layout (MyCoordinates): positions of output speaker layout: (N speakers)

    Returns:
        intensity (jax.numpy array): intensity vector for each virtual source,
                cartesian coordinates (Lx3)
    """

    # Energy calculation
    energy = energy_calculation(speaker_signals)

    # Cardinal coordinates - unitary vectors of output speakers - Ui
    U = output_layout.cart()

    # Intensity
    aux = jnp.dot(speaker_signals**2, U)
    intensity = jnp.array((1 / (energy + jnp.finfo(float).eps)).reshape(-1, 1) * aux)

    return intensity


def radial_I_calculation(
    cloud_points: MyCoordinates,
    speaker_signals: jnp,
    output_layout: MyCoordinates,
):
    """Function to obtain the radial intensity of each virtual source of a cloud of points in an output
    layout out from the coded channel gains

    Args:
        cloud_points (MyCoordinates): position of the virtual sources pyfar.Coordinates (L sources)
        speaker_signals (numpy Array): speaker signals resulting from decoding 
                to input set of encoded L directions (LxN size)
        output_layout (MyCoordinates): positions of output speaker layout: (N speakers)

    Returns:
        radial_intensity (jax.numpy array): contains the radial intensity values for each virtual
                source (1xL)
    """
    # Intensity calculation
    intensity = intensity_calculation(speaker_signals, output_layout)

    # Cardinal coordinates direction virtual source - Vj
    V = cloud_points.cart()

    # Scalar product of intensity and V
    radial_intensity = jnp.sum(jnp.multiply(intensity, V), axis=1)

    return radial_intensity


def transverse_I_calculation(
    cloud_points: MyCoordinates,
    speaker_signals: jnp,
    output_layout: MyCoordinates,
):
    """Function to obtain the transverse intensity of each virtual source of a cloud of points in an output
    layout out from the coded channel gains

    Args:
        cloud_points (MyCoordinates): position of the virtual sources pyfar.Coordinates (L sources)
        speaker_signals (numpy Array): speaker signals resulting from decoding 
                to input set of encoded L directions (LxN size)
        output_layout (MyCoordinates): positions of output speaker layout: (N speakers)

    Returns:
        transversal_intensity (jax.numpy array): contains the transversal intensity values for each virtual
                source (1xL)
    """
    # Intensity calculation
    intensity = intensity_calculation(speaker_signals, output_layout)

    # Cardinal coordinates direction virtual source - Vj
    V = cloud_points.cart()

    # Vectorial product of intensity and V
    transverse_intensity = jnp.array(
        jnp.linalg.norm(jnp.cross(intensity, V, axis=1), axis=1)
    )

    return transverse_intensity


def angular_error(radial: jnp, transverse: jnp):
    """Function to obtain the angular error of each virtual source of a cloud of points given the
    radial and transverse components of the intensity

    Args:
        radial (jax.numpy array): contains the radial intensity values for each virtual source (1xL)
        transverse (jax.numpy array): contains the transversal intensity values for each virtual source (1xL)

    Returns:
        error (jax.numpy array): contains the angular error values for each virtual source (1xL)
    """
    tan = jnp.abs(transverse) / jnp.abs(radial)
    error_rad = jnp.arctan(tan)  # -pi/2 to pi/2
    error = jnp.degrees(error_rad)  # to degrees
    return error


def width_angle(intensity: jnp):
    """Function to obtain the width of each virtual source of a cloud of points given the
    radial component of the intensity

    Args:
        intensity (jax.numpy array): contains the radial intensity values for each virtual source (1xL)

    Returns:
        width (jax.numpy array): contains the width values for each virtual source (1xL) in degrees
    """
    width = jnp.degrees(jnp.arccos(jnp.abs(intensity)))
    return width
