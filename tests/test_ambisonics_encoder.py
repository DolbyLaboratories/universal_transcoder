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
from universal_transcoder.encoders.ambisonics_encoder import ambisonics_encoder


def test_ambisonics_encoder1():
    virtual = MyCoordinates.point(0, 0)
    n = 1
    wxyz = ambisonics_encoder(virtual, n)
    expected = np.array([1, 0, 0, 1])
    np.testing.assert_allclose(wxyz, expected, atol=1e-15)


def test_ambisonics_encoder2():
    virtual = MyCoordinates.point(90, 0)
    n = 2
    wxyz = ambisonics_encoder(virtual, n)
    expected = np.array([1, 1, 0, 0, 0, 0, -0.5, 0, -0.8660254038])
    np.testing.assert_allclose(wxyz, expected, atol=1e-15)  # rtol=1e-07


def test_ambisonics_encoder3():
    virtual = MyCoordinates.point(90, 0)
    n = 4
    size = ambisonics_encoder(virtual, n).size
    expected = (n + 1) ** 2
    np.testing.assert_allclose(size, expected, atol=1e-15)  # rtol=1e-07


def test_ambisonics_encoder4():
    virtual = MyCoordinates.point(90, 0)
    n = 5
    size = ambisonics_encoder(virtual, n).size
    expected = (n + 1) ** 2
    np.testing.assert_allclose(size, expected, atol=1e-15)  # rtol=1e-07
