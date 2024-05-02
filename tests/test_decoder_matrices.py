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

from universal_transcoder.auxiliars.get_decoder_matrices import get_vbap_decoder_matrix
from universal_transcoder.auxiliars.my_coordinates import MyCoordinates


def test_decoder_matrices1():  # VBAP 2D-->2D
    layout_input = MyCoordinates.mult_points(
        np.array([(-90, 0, 1), (90, 0, 1), (180, 0, 1)])
    )
    layout_output = MyCoordinates.mult_points(
        np.array([(-90, 0, 1), (0, 0, 1), (90, 0, 1), (180, 0, 1)])
    )
    expected = np.array([(1, 0, 0), (0, 0, 0), (0, 1, 0), (0, 0, 1)])

    result = get_vbap_decoder_matrix(layout_input, layout_output)
    print(result)
    np.testing.assert_allclose(result, expected, atol=1e-15)


def test_decoder_matrices2():  # VBAP 3D-->3D
    layout_input = MyCoordinates.mult_points(
        np.array(
            [
                (-130, 0, 1),
                (-40, 45, 1),
                (-25, 0, 1),
                (0, 0, 1),
                (25, 0, 1),
                (130, 0, 1),
            ]
        )
    )
    layout_output = MyCoordinates.mult_points(
        np.array(
            [
                (-130, 0, 1),
                (-50, 0, 1),
                (-40, 45, 1),
                (0, 0, 1),
                (40, 45, 1),
                (50, 0, 1),
                (130, 0, 1),
                (180, 45, 1),
            ]
        )
    )

    expected = np.array(
        [
            (1, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 1, 0, 0, 0, 0, 0),
            (0, 0.5, 0, 0.5, 0, 0, 0, 0),
            (0, 0, 0, 1, 0, 0, 0, 0),
            (0, 0, 0, 0.5, 0, 0.5, 0, 0),
            (0, 0, 0, 0, 0, 0, 1, 0),
        ]
    ).T

    result = get_vbap_decoder_matrix(layout_input, layout_output)
    np.testing.assert_allclose(result, expected, atol=1e-15)


def test_decoder_matrices3():  # VBAP 3D-->2D
    layout_input = MyCoordinates.mult_points(
        np.array([(-90, 0, 1), (-90, 45, 1), (0, 0, 1), (90, 0, 1), (90, 45, 1)])
    )
    layout_output = MyCoordinates.mult_points(
        np.array([(-90, 0, 1), (0, 0, 1), (90, 0, 1)])
    )

    expected = np.array([(1, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 1)]).T

    result = get_vbap_decoder_matrix(layout_input, layout_output)
    print(result)
    np.testing.assert_allclose(result, expected, atol=1e-15)


def test_decoder_matrices4():  # VBAP 2D-->3D
    layout_input = MyCoordinates.mult_points(
        np.array([(-90, 0, 1), (0, 0, 1), (90, 0, 1), (180, 0, 1)])
    )
    layout_output = MyCoordinates.mult_points(
        np.array(
            [(-90, 0, 1), (-90, 45, 1), (0, 0, 1), (90, 0, 1), (90, 45, 1), (180, 0, 1)]
        )
    )

    expected = np.array(
        [(1, 0, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0), (0, 0, 0, 1, 0, 0), (0, 0, 0, 0, 0, 1)]
    ).T

    result = get_vbap_decoder_matrix(layout_input, layout_output)
    np.testing.assert_allclose(result, expected, atol=1e-15)
