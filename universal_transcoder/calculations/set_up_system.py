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
import numpy as np

from universal_transcoder.calculations.cost_function import State
from ..auxiliars.get_decoder_matrices import get_ambisonics_decoder_matrix


def set_up_general(info: dict):
    """Function to prepare the system for optimization. It generates the initial decoding matrix
    of the appropiate size and it stores in class State all the needed data for the optimization

    Args:
        info (dict): All the info needed for the optimization given as the following:
            dictionary = {
                "input_matrix_optimization": input,     # Input matrix that encodes in input format LxM
                "cloud_optimization": cloud,            # Cloud of points sampling the sphere (L)
                "output_layout": output_layout,         # Output layout of speakers to decode
                "coefficients": {                       # List of coefficients to the cost function
                    "energy": 0,
                    "radial_intensity": 0,
                    "transverse_intensity": 0,
                    "pressure": 0,
                    "radial_velocity": 0,
                    "transverse_velocity": 0,
                    "in_phase_lin": 0,
                    "symmetry_lin": 0,
                    "total_gains_quad": 0,
                    "in_phase_quad": 0,
                    "symmetry_quad": 0,
                    "total_gains_quad": 0,
                },
                "directional_weights": 1,               # Weights given to different directions sampling the sphere
                "show_results": True,                   # Flag to show results
                "save_results": False,                  # Flag to save results
                "cloud_plots": cloud,                   # Cloud of points sampling the sphere (P) for plots
                "input_matrix_plots": matrix,           # Matrix that encodes in input format PxM, for plots
                "results_file_name": "name",            # String of folder name where to save, if save_results=True
            }

    Returns:
        current_state (State): contains side-information that the optimization process needs (1xL)
        T_flatten_initial: (jax.numpy Array): vector of shape 1xNM to optimize
    """
    cloud = info["cloud_optimization"]
    layout = info["output_layout"]
    input_matrix = info["input_matrix_optimization"]

    M = input_matrix.shape[1]
    N = layout.sph_deg().shape[0]

    Dspk = 1
    if "transcoding" in info.keys() and info["transcoding"]:
        # If "transcoding" is activated
        # because in case transcoding is activated, the transcoding is to Ambisonics
        #  FIXME this should be more general
        #  FIXME it should have the possibility to choose between 2D and 3D Ambisonics (at least)
        order = info["ambisonics_encoding_order"]
        N = (order + 1) ** 2
        Dspk = get_ambisonics_decoder_matrix(order, layout, "pseudo")

    # Generate Decoding initial matrix
    np.random.seed(0)
    save_shape = (N, M)
    T_flatten_initial = jnp.ones(N * M) + 0.2 * jnp.array(np.random.randn(N * M))

    # Create 'current state' class
    current_state = State(cloud, input_matrix, layout, save_shape, T_flatten_initial)

    # Set coefficients
    # Quadratic terms
    current_state.ce = info["coefficients"]["energy"]
    current_state.cir = info["coefficients"]["radial_intensity"]
    current_state.cit = info["coefficients"]["transverse_intensity"]
    current_state.cphquad = info["coefficients"]["in_phase_quad"]
    current_state.csymquad = info["coefficients"]["symmetry_quad"]
    current_state.cmingquad = info["coefficients"]["total_gains_quad"]
    # Linear terms
    current_state.cp = info["coefficients"]["pressure"]
    current_state.cvr = info["coefficients"]["radial_velocity"]
    current_state.cvt = info["coefficients"]["transverse_velocity"]
    current_state.cphlin = info["coefficients"]["in_phase_lin"]
    current_state.csymlin = info["coefficients"]["symmetry_lin"]
    current_state.cminglin = info["coefficients"]["total_gains_lin"]

    # Others
    current_state.static_decoding_matrix = Dspk

    # Set weights
    current_state.w = info["directional_weights"]

    # Calculate cost with initial decoding matrix
    initial_cost = current_state.cost_function(T_flatten_initial)
    print("\nCost value for initial decoding matrix: ", initial_cost)

    return current_state, T_flatten_initial
