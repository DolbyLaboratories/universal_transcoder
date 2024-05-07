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

import math as mh

import numpy as np


def get_convention(conventions: str, n: int):
    """Function to obtain the adapting matrix from one convention to another
    For now, only N3D to SN3D

    Args:
        conventions (string): conventions from and towards we desire
        n (int): desired ambisonics order

    Returns:
        adapter (numpy array): adapting matrix
    """

    adapter = []
    if conventions == "n3d2sn3d":
        # 0 - order
        if n >= 0:
            aux = 1 / mh.sqrt(2 * 0 + 1)
            adapter.append(aux)  # W

        # 1 - order
        if n >= 1:
            aux = 1 / mh.sqrt(2 * 1 + 1)
            adapter.append(aux)  # Y
            adapter.append(aux)  # Z
            adapter.append(aux)  # X

        # 2 - order
        if n >= 2:
            aux = 1 / mh.sqrt(2 * 2 + 1)
            adapter.append(aux)  # V
            adapter.append(aux)  # T
            adapter.append(aux)  # R
            adapter.append(aux)  # S
            adapter.append(aux)  # U

        # 3 - order
        if n >= 3:
            aux = 1 / mh.sqrt(2 * 3 + 1)
            adapter.append(aux)  # Q
            adapter.append(aux)  # O
            adapter.append(aux)  # M
            adapter.append(aux)  # K
            adapter.append(aux)  # L
            adapter.append(aux)  # N
            adapter.append(aux)  # P

        # 4 - order
        if n >= 4:
            aux = 1 / mh.sqrt(2 * 4 + 1)
            adapter.append(aux)  # (4,-4)
            adapter.append(aux)  # (4,-3)
            adapter.append(aux)  # (4,-2)
            adapter.append(aux)  # (4,-1)
            adapter.append(aux)  # (4,-0)
            adapter.append(aux)  # (4, 1)
            adapter.append(aux)  # (4, 2)
            adapter.append(aux)  # (4, 3)
            adapter.append(aux)  # (4, 4)

        # 5 - order
        if n >= 5:
            aux = 1 / mh.sqrt(2 * 5 + 1)
            adapter.append(aux)  # (5,-5)
            adapter.append(aux)  # (5,-4)
            adapter.append(aux)  # (5, -3)
            adapter.append(aux)  # (5,-2)
            adapter.append(aux)  # (5,-1)
            adapter.append(aux)  # (5, 0)
            adapter.append(aux)  # (5, 1)
            adapter.append(aux)  # (5, 2)
            adapter.append(aux)  # (5, 3)
            adapter.append(aux)  # (5, 4)
            adapter.append(aux)  # (5, 5)
    else:
        raise ValueError(f"Unsupported convention '{conventions}'")

    return np.asarray(adapter)
