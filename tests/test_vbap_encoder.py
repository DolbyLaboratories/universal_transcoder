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

from universal_transcoder.auxiliars.my_coordinates import MyCoordinates
from universal_transcoder.encoders.vbap_encoder import vbap_2D_encoder
from universal_transcoder.encoders.vbap_encoder import vbap_3D_encoder


# VBAP 3D
def test_vbap_3D_encoder1():
    virtual = MyCoordinates.point(0, 0)
    layout = MyCoordinates.mult_points(
        np.array(
            [
                (-130, 0, 1),
                (-120, 45, 1),
                (-50, 0, 1),
                (-40, 45, 1),
                (0, 0, 1),
                (40, 45, 1),
                (50, 0, 1),
                (120, 45, 1),
                (130, 0, 1),
            ]
        )
    )
    speaker_gains = vbap_3D_encoder(virtual, layout)
    expected_gains = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0])
    np.testing.assert_allclose(speaker_gains, expected_gains, atol=1e-15)


def test_vbap_3D_encoder2():
    virtual = MyCoordinates.point(-25, 0)
    layout = MyCoordinates.mult_points(
        np.array(
            [
                (-130, 0, 1),
                (-120, 45, 1),
                (-50, 0, 1),
                (-40, 45, 1),
                (0, 0, 1),
                (40, 45, 1),
                (50, 0, 1),
                (120, 45, 1),
                (130, 0, 1),
            ]
        )
    )
    speaker_gains = vbap_3D_encoder(virtual, layout)
    expected_gains = np.array([0, 0, 0.5, 0, 0.5, 0, 0, 0, 0])
    np.testing.assert_allclose(speaker_gains, expected_gains, atol=1e-15)


def test_vbap_3D_encoder3():
    virtual = MyCoordinates.point(120, 45)
    layout = MyCoordinates.mult_points(
        np.array(
            [
                (-130, 0, 1),
                (-120, 45, 1),
                (-50, 0, 1),
                (-40, 45, 1),
                (0, 0, 1),
                (40, 45, 1),
                (50, 0, 1),
                (120, 45, 1),
                (130, 0, 1),
            ]
        )
    )
    speaker_gains = vbap_3D_encoder(virtual, layout)
    expected_gains = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0])
    np.testing.assert_allclose(speaker_gains, expected_gains, atol=1e-15)


def test_vbap_3D_encoder4():
    layout = MyCoordinates.mult_points(
        np.array(
            [
                (-130, 0, 1),
                (-120, 45, 1),
                (-50, 0, 1),
                (-40, 45, 1),
                (0, 0, 1),
                (40, 45, 1),
                (50, 0, 1),
                (120, 45, 1),
                (130, 0, 1),
            ]
        )
    )
    virtual = MyCoordinates.mult_points_cart(
        np.mean(layout.cart()[:3, :], axis=0, keepdims=True)
    )
    speaker_gains = vbap_3D_encoder(virtual, layout)
    expected_gains = np.array([1 / 3, 1 / 3, 1 / 3, 0, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(speaker_gains, expected_gains, atol=1e-15)


# VBAP 2D
def test_vbap_2D_encoder1():
    virtual = MyCoordinates.point(120, 0)
    layout = MyCoordinates.mult_points(
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
    speaker_gains = vbap_2D_encoder(virtual, layout)
    expected_gains = np.array([0, 0, 0, 0, 0, 1, 0])
    np.testing.assert_allclose(speaker_gains, expected_gains, atol=1e-15)


def test_vbap_2D_encoder2():
    virtual = MyCoordinates.point(25, 0)
    layout = MyCoordinates.mult_points(
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
    speaker_gains = vbap_2D_encoder(virtual, layout)
    expected_gains = np.array([0, 0, 0, 0.5, 0.5, 0, 0])
    np.testing.assert_allclose(speaker_gains, expected_gains, atol=1e-15)


def test_vbap_2D_encoder3():
    virtual = MyCoordinates.point(180, 0)
    layout = MyCoordinates.mult_points(
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
    speaker_gains = vbap_2D_encoder(virtual, layout)
    expected_gains = np.array([0.5, 0, 0, 0, 0, 0, 0.5])
    np.testing.assert_allclose(speaker_gains, expected_gains, atol=1e-15)
