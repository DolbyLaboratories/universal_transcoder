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

import numpy as np
from universal_transcoder.auxiliars.get_decoder_matrices import (
    get_vbap_decoder_matrix,
    get_ambisonics_decoder_matrix,
)
from universal_transcoder.auxiliars.get_input_channels import (
    get_input_channels_ambisonics,
    get_input_channels_vbap,
)
from universal_transcoder.auxiliars.my_coordinates import MyCoordinates
from universal_transcoder.calculations.energy_intensity import (
    energy_calculation,
    radial_I_calculation,
    transverse_I_calculation,
)
from universal_transcoder.calculations.pressure_velocity import (
    pressure_calculation,
    radial_V_calculation,
)


# VBAP single point - check values


def test_energy1():
    # VBAP 2D testing
    virtual = MyCoordinates.point(45, 0)
    input_layout = MyCoordinates.mult_points(
        np.array([(-90, 0, 1), (45, 0, 1), (90, 0, 1), (180, 0, 1)])
    )  # 2D square
    output_layout = MyCoordinates.mult_points(
        np.array(
            [(-90, 0, 1), (0, 90, 1), (0, -90, 1), (45, 0, 1), (90, 0, 1), (180, 0, 1)]
        )
    )  # 3D tetraedo
    decoder_matrix = get_vbap_decoder_matrix(input_layout, output_layout)
    input_channels = get_input_channels_vbap(virtual, input_layout)

    energy = energy_calculation(input_channels, decoder_matrix)

    expected = 1
    np.testing.assert_allclose(energy, expected, atol=1e-15)


def test_Irad1():
    # VBAP 2D testing
    virtual = MyCoordinates.point(45, 0)
    input_layout = MyCoordinates.mult_points(
        np.array([(-90, 0, 1), (45, 0, 1), (90, 0, 1), (180, 0, 1)])
    )  # 2D square
    output_layout = MyCoordinates.mult_points(
        np.array(
            [(-90, 0, 1), (0, 90, 1), (0, -90, 1), (45, 0, 1), (90, 0, 1), (180, 0, 1)]
        )
    )  # 3D tetraedo
    decoder_matrix = get_vbap_decoder_matrix(input_layout, output_layout)
    input_channels = get_input_channels_vbap(virtual, input_layout)

    rad = radial_I_calculation(virtual, input_channels, output_layout, decoder_matrix)

    expected = 1
    np.testing.assert_allclose(rad, expected, atol=1e-15)


def test_Itrans1():
    # VBAP 2D testing
    virtual = MyCoordinates.point(45, 0)
    input_layout = MyCoordinates.mult_points(
        np.array([(-90, 0, 1), (45, 0, 1), (90, 0, 1), (180, 0, 1)])
    )  # 2D square
    output_layout = MyCoordinates.mult_points(
        np.array(
            [(-90, 0, 1), (0, 90, 1), (0, -90, 1), (45, 0, 1), (90, 0, 1), (180, 0, 1)]
        )
    )  # 3D tetraedo
    decoder_matrix = get_vbap_decoder_matrix(input_layout, output_layout)
    input_channels = get_input_channels_vbap(virtual, input_layout)

    trans = transverse_I_calculation(
        virtual, input_channels, output_layout, decoder_matrix
    )

    expected = 0
    np.testing.assert_allclose(trans, expected, atol=1e-07)


def test_Pres1():
    # VBAP 2D testing
    virtual = MyCoordinates.point(45, 0)
    input_layout = MyCoordinates.mult_points(
        np.array([(-90, 0, 1), (45, 0, 1), (90, 0, 1), (180, 0, 1)])
    )  # 2D square
    output_layout = MyCoordinates.mult_points(
        np.array(
            [(-90, 0, 1), (0, 90, 1), (0, -90, 1), (45, 0, 1), (90, 0, 1), (180, 0, 1)]
        )
    )  # 3D tetraedo
    decoder_matrix = get_vbap_decoder_matrix(input_layout, output_layout)
    input_channels = get_input_channels_vbap(virtual, input_layout)

    pressure = pressure_calculation(input_channels, decoder_matrix)

    expected = 1
    np.testing.assert_allclose(pressure, expected, atol=1e-07)


def test_Vel1():
    # VBAP 2D testing
    virtual = MyCoordinates.point(45, 0)
    input_layout = MyCoordinates.mult_points(
        np.array([(-90, 0, 1), (45, 0, 1), (90, 0, 1), (180, 0, 1)])
    )  # 2D square
    output_layout = MyCoordinates.mult_points(
        np.array(
            [(-90, 0, 1), (0, 90, 1), (0, -90, 1), (45, 0, 1), (90, 0, 1), (180, 0, 1)]
        )
    )  # 3D tetraedo
    decoder_matrix = get_vbap_decoder_matrix(input_layout, output_layout)
    input_channels = get_input_channels_vbap(virtual, input_layout)

    radial_v = radial_V_calculation(
        virtual, input_channels, output_layout, decoder_matrix
    )

    expected = 1
    np.testing.assert_allclose(radial_v, expected, atol=1e-07)


# AMBISONICS single point - check values


def test_energy2():
    virtual = MyCoordinates.point(45, 0)
    order = 3
    output_layout = MyCoordinates.mult_points(
        np.array(
            [
                (-135, 0, 1),
                (-45, 0, 1),
                (0, 90, 1),
                (0, -90, 1),
                (45, 0, 1),
                (135, 0, 1),
            ]
        )
    )  # 3D tetraedo
    decoder_matrix = get_ambisonics_decoder_matrix(order, output_layout, "pseudo")
    input_channels = get_input_channels_ambisonics(virtual, order)

    energy = energy_calculation(input_channels, decoder_matrix)
    print(energy)

    expected = 1
    np.testing.assert_allclose(energy, expected, atol=1e-07)


def test_Irad2():
    virtual = MyCoordinates.point(45, 0)
    order = 3
    output_layout = MyCoordinates.mult_points(
        np.array(
            [
                (-135, 0, 1),
                (-45, 0, 1),
                (0, 90, 1),
                (0, -90, 1),
                (45, 0, 1),
                (135, 0, 1),
            ]
        )
    )  # 3D tetraedo
    decoder_matrix = get_ambisonics_decoder_matrix(order, output_layout, "pseudo")
    input_channels = get_input_channels_ambisonics(virtual, order)

    rad = radial_I_calculation(virtual, input_channels, output_layout, decoder_matrix)
    print(rad)

    expected = 1
    np.testing.assert_allclose(rad, expected, atol=1e-15)


def test_Itrans2():
    virtual = MyCoordinates.point(45, 0)
    order = 3
    output_layout = MyCoordinates.mult_points(
        np.array(
            [
                (-135, 0, 1),
                (-45, 0, 1),
                (0, 90, 1),
                (0, -90, 1),
                (45, 0, 1),
                (135, 0, 1),
            ]
        )
    )  # 3D tetraedo
    decoder_matrix = get_ambisonics_decoder_matrix(order, output_layout, "pseudo")
    input_channels = get_input_channels_ambisonics(virtual, order)

    trans = transverse_I_calculation(
        virtual, input_channels, output_layout, decoder_matrix
    )
    print(trans)

    expected = 0
    np.testing.assert_allclose(trans, expected, atol=1e-07)


def test_Pres2():
    # Ambisonics 3D regular layout testing - PSEUDO
    virtual = MyCoordinates.point(45, 0)
    order = 3
    output_layout = MyCoordinates.mult_points(
        np.array(
            [
                (-135, 0, 1),
                (-45, 0, 1),
                (0, 90, 1),
                (0, -90, 1),
                (45, 0, 1),
                (135, 0, 1),
            ]
        )
    )  # 3D tetraedo
    decoder_matrix = get_ambisonics_decoder_matrix(order, output_layout, "pseudo")
    input_channels = get_input_channels_ambisonics(virtual, order)

    pressure = pressure_calculation(input_channels, decoder_matrix)

    expected = 1
    np.testing.assert_allclose(pressure, expected, atol=1e-07)


def test_Vel2():
    # Ambisonics 3D regular layout testing - PSEUDO
    virtual = MyCoordinates.point(45, 0)
    order = 3
    output_layout = MyCoordinates.mult_points(
        np.array(
            [
                (-135, 0, 1),
                (-45, 0, 1),
                (0, 90, 1),
                (0, -90, 1),
                (45, 0, 1),
                (135, 0, 1),
            ]
        )
    )  # 3D tetraedo
    decoder_matrix = get_ambisonics_decoder_matrix(order, output_layout, "pseudo")
    input_channels = get_input_channels_ambisonics(virtual, order)

    radial_v = radial_V_calculation(
        virtual, input_channels, output_layout, decoder_matrix
    )

    expected = 1
    np.testing.assert_allclose(radial_v, expected, atol=1e-07)


def test_Pres3():
    # Ambisonics 3D regular layout testing - NORM
    virtual = MyCoordinates.point(45, 0)
    order = 3
    output_layout = MyCoordinates.mult_points(
        np.array(
            [
                (-135, 0, 1),
                (-45, 0, 1),
                (0, 90, 1),
                (0, -90, 1),
                (45, 0, 1),
                (135, 0, 1),
            ]
        )
    )  # 3D tetraedo
    decoder_matrix = get_ambisonics_decoder_matrix(order, output_layout, "norm")
    input_channels = get_input_channels_ambisonics(virtual, order)

    pressure = pressure_calculation(input_channels, decoder_matrix)

    expected = 1
    np.testing.assert_allclose(pressure, expected, atol=1e-07)


# VBAP testing cloud of points - check vector sizes
def test_energy4():
    # VBAP 2D testing
    output_layout = MyCoordinates.mult_points(
        np.array(
            [
                (-130, 0, 1),
                (-120, 0, 1),
                (-50, 0, 1),
                (0, 0, 1),
                (50, 0, 1),
                (120, 0, 1),
                (130, 0, 1),
            ]
        )
    )
    input_layout = MyCoordinates.mult_points(
        np.array([(-120, 0, 1), (-50, 0, 1), (0, 0, 1), (50, 0, 1), (120, 0, 1)])
    )
    cloud = MyCoordinates.mult_points(
        np.array([(-50, 0, 1), (0, 0, 1), (50, 0, 1), (100, 0, 1)])
    )

    decoder_matrix = get_vbap_decoder_matrix(input_layout, output_layout)
    input_channels = get_input_channels_vbap(cloud, input_layout)

    energy = energy_calculation(input_channels, decoder_matrix)

    expected = cloud.cart().shape[0]
    np.testing.assert_allclose(energy.shape[0], expected, atol=1e-15)


def test_Irad4():
    # VBAP 2D testing
    output_layout = MyCoordinates.mult_points(
        np.array(
            [
                (-130, 0, 1),
                (-120, 0, 1),
                (-50, 0, 1),
                (0, 0, 1),
                (50, 0, 1),
                (120, 0, 1),
                (130, 0, 1),
            ]
        )
    )
    input_layout = MyCoordinates.mult_points(
        np.array([(-120, 0, 1), (-50, 0, 1), (0, 0, 1), (50, 0, 1), (120, 0, 1)])
    )
    cloud = MyCoordinates.mult_points(
        np.array([(-50, 0, 1), (0, 0, 1), (50, 0, 1), (100, 0, 1)])
    )

    decoder_matrix = get_vbap_decoder_matrix(input_layout, output_layout)
    input_channels = get_input_channels_vbap(cloud, input_layout)

    rad = radial_I_calculation(cloud, input_channels, output_layout, decoder_matrix)

    expected = cloud.cart().shape[0]
    np.testing.assert_allclose(rad.shape[0], expected, atol=1e-15)


def test_Itrans4():
    # VBAP 2D testing
    output_layout = MyCoordinates.mult_points(
        np.array(
            [
                (-130, 0, 1),
                (-120, 0, 1),
                (-50, 0, 1),
                (0, 0, 1),
                (50, 0, 1),
                (120, 0, 1),
                (130, 0, 1),
            ]
        )
    )
    input_layout = MyCoordinates.mult_points(
        np.array([(-120, 0, 1), (-50, 0, 1), (0, 0, 1), (50, 0, 1), (120, 0, 1)])
    )
    cloud = MyCoordinates.mult_points(
        np.array([(-50, 0, 1), (0, 0, 1), (50, 0, 1), (100, 0, 1)])
    )

    decoder_matrix = get_vbap_decoder_matrix(input_layout, output_layout)
    input_channels = get_input_channels_vbap(cloud, input_layout)

    trans = transverse_I_calculation(
        cloud, input_channels, output_layout, decoder_matrix
    )

    expected = cloud.cart().shape[0]
    np.testing.assert_allclose(trans.shape[0], expected, atol=1e-07)


def test_Pres4():
    # VBAP 2D testing
    output_layout = MyCoordinates.mult_points(
        np.array(
            [
                (-130, 0, 1),
                (-120, 0, 1),
                (-50, 0, 1),
                (0, 0, 1),
                (50, 0, 1),
                (120, 0, 1),
                (130, 0, 1),
            ]
        )
    )
    input_layout = MyCoordinates.mult_points(
        np.array([(-120, 0, 1), (-50, 0, 1), (0, 0, 1), (50, 0, 1), (120, 0, 1)])
    )
    cloud = MyCoordinates.mult_points(
        np.array([(-50, 0, 1), (0, 0, 1), (50, 0, 1), (100, 0, 1)])
    )

    decoder_matrix = get_vbap_decoder_matrix(input_layout, output_layout)
    input_channels = get_input_channels_vbap(cloud, input_layout)

    pressure = pressure_calculation(input_channels, decoder_matrix)

    expected = cloud.cart().shape[0]
    np.testing.assert_allclose(pressure.shape[0], expected, atol=1e-07)


def test_Vel4():
    # VBAP 2D testing
    output_layout = MyCoordinates.mult_points(
        np.array(
            [
                (-130, 0, 1),
                (-120, 0, 1),
                (-50, 0, 1),
                (0, 0, 1),
                (50, 0, 1),
                (120, 0, 1),
                (130, 0, 1),
            ]
        )
    )
    input_layout = MyCoordinates.mult_points(
        np.array([(-120, 0, 1), (-50, 0, 1), (0, 0, 1), (50, 0, 1), (120, 0, 1)])
    )
    cloud = MyCoordinates.mult_points(
        np.array([(-50, 0, 1), (0, 0, 1), (50, 0, 1), (100, 0, 1)])
    )

    decoder_matrix = get_vbap_decoder_matrix(input_layout, output_layout)
    input_channels = get_input_channels_vbap(cloud, input_layout)

    radial_v = radial_V_calculation(
        cloud, input_channels, output_layout, decoder_matrix
    )

    expected = cloud.cart().shape[0]
    np.testing.assert_allclose(radial_v.shape[0], expected, atol=1e-07)


# AMBISONICS testing cloud of points - check vector sizes


def test_energy5():
    output_layout = MyCoordinates.mult_points(
        np.array(
            [
                (-130, 0, 1),
                (-120, 0, 1),
                (-50, 0, 1),
                (0, 0, 1),
                (50, 0, 1),
                (120, 0, 1),
                (130, 0, 1),
            ]
        )
    )
    cloud = MyCoordinates.mult_points(
        np.array([(-50, 0, 1), (0, 0, 1), (50, 0, 1), (100, 0, 1)])
    )
    order = 1

    decoder_matrix = get_ambisonics_decoder_matrix(order, output_layout, "pseudo")
    input_channels = get_input_channels_ambisonics(cloud, order)

    energy = energy_calculation(input_channels, decoder_matrix)

    expected = cloud.cart().shape[0]
    np.testing.assert_allclose(energy.shape[0], expected, atol=1e-07)


def test_Irad5():
    output_layout = MyCoordinates.mult_points(
        np.array(
            [
                (-130, 0, 1),
                (-120, 0, 1),
                (-50, 0, 1),
                (0, 0, 1),
                (50, 0, 1),
                (120, 0, 1),
                (130, 0, 1),
            ]
        )
    )
    cloud = MyCoordinates.mult_points(
        np.array([(-50, 0, 1), (0, 0, 1), (50, 0, 1), (100, 0, 1)])
    )
    order = 1

    decoder_matrix = get_ambisonics_decoder_matrix(order, output_layout, "pseudo")
    input_channels = get_input_channels_ambisonics(cloud, order)

    rad = radial_I_calculation(cloud, input_channels, output_layout, decoder_matrix)

    expected = cloud.cart().shape[0]
    np.testing.assert_allclose(rad.shape[0], expected, atol=1e-15)


def test_Itrans5():
    output_layout = MyCoordinates.mult_points(
        np.array(
            [
                (-130, 0, 1),
                (-120, 0, 1),
                (-50, 0, 1),
                (0, 0, 1),
                (50, 0, 1),
                (120, 0, 1),
                (130, 0, 1),
            ]
        )
    )
    cloud = MyCoordinates.mult_points(
        np.array([(-50, 0, 1), (0, 0, 1), (50, 0, 1), (100, 0, 1)])
    )
    order = 1

    decoder_matrix = get_ambisonics_decoder_matrix(order, output_layout, "pseudo")
    input_channels = get_input_channels_ambisonics(cloud, order)

    trans = transverse_I_calculation(
        cloud, input_channels, output_layout, decoder_matrix
    )

    expected = cloud.cart().shape[0]
    np.testing.assert_allclose(trans.shape[0], expected, atol=1e-07)


def test_Pres5():
    output_layout = MyCoordinates.mult_points(
        np.array(
            [
                (-130, 0, 1),
                (-120, 0, 1),
                (-50, 0, 1),
                (0, 0, 1),
                (50, 0, 1),
                (120, 0, 1),
                (130, 0, 1),
            ]
        )
    )
    cloud = MyCoordinates.mult_points(
        np.array([(-50, 0, 1), (0, 0, 1), (50, 0, 1), (100, 0, 1)])
    )
    order = 1

    decoder_matrix = get_ambisonics_decoder_matrix(order, output_layout, "pseudo")
    input_channels = get_input_channels_ambisonics(cloud, order)

    pressure = pressure_calculation(input_channels, decoder_matrix)

    expected = cloud.cart().shape[0]
    np.testing.assert_allclose(pressure.shape[0], expected, atol=1e-07)


def test_Vel5():
    output_layout = MyCoordinates.mult_points(
        np.array(
            [
                (-130, 0, 1),
                (-120, 0, 1),
                (-50, 0, 1),
                (0, 0, 1),
                (50, 0, 1),
                (120, 0, 1),
                (130, 0, 1),
            ]
        )
    )
    cloud = MyCoordinates.mult_points(
        np.array([(-50, 0, 1), (0, 0, 1), (50, 0, 1), (100, 0, 1)])
    )
    order = 1

    decoder_matrix = get_ambisonics_decoder_matrix(order, output_layout, "pseudo")
    input_channels = get_input_channels_ambisonics(cloud, order)

    radial_v = radial_V_calculation(
        cloud, input_channels, output_layout, decoder_matrix
    )

    expected = cloud.cart().shape[0]
    np.testing.assert_allclose(radial_v.shape[0], expected, atol=1e-07)


def test_Pres6():
    output_layout = MyCoordinates.mult_points(
        np.array(
            [
                (-130, 0, 1),
                (-120, 0, 1),
                (-50, 0, 1),
                (0, 0, 1),
                (50, 0, 1),
                (120, 0, 1),
                (130, 0, 1),
            ]
        )
    )
    cloud = MyCoordinates.mult_points(
        np.array([(-50, 0, 1), (0, 0, 1), (50, 0, 1), (100, 0, 1)])
    )
    order = 1

    decoder_matrix = get_ambisonics_decoder_matrix(order, output_layout, "norm")
    input_channels = get_input_channels_ambisonics(cloud, order)

    pressure = pressure_calculation(input_channels, decoder_matrix)

    expected = cloud.cart().shape[0]
    np.testing.assert_allclose(pressure.shape[0], expected, atol=1e-07)
