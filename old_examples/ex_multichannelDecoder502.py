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

from pathlib import Path

import numpy as np

from universal_transcoder.auxiliars.get_cloud_points import (
    get_equi_circumference_points,
    get_equi_t_design_points,
    mix_clouds_of_points,
    get_all_sphere_points,
)
from universal_transcoder.auxiliars.get_input_channels import (
    get_input_channels_vbap,
)
from universal_transcoder.auxiliars.my_coordinates import MyCoordinates
from universal_transcoder.plots_and_logs.all_plots import plots_general
from universal_transcoder.calculations.optimization import optimize


# USAD #######################################################


# Set of output speakers
output_layout = MyCoordinates.mult_points(
    np.array(
        [
            (30, 0, 1),
            (-30, 0, 1),
            (0, 0, 1),
            (110, 0, 1),
            (-110, 0, 1),
            (90, 45, 1),
            (-90, 45, 1),
        ]
    )
)

# Cloud of points to be encoded in input format (one-to-one panning)
cloud_optimization = get_all_sphere_points(10, plot_show=False).discard_lower_hemisphere()

# Input matrix for optimization
n = cloud_optimization.cart().shape[0]
input_matrix_optimization = np.identity(n)



dictionary = {
    "input_matrix_optimization": input_matrix_optimization,
    "cloud_optimization": cloud_optimization,
    "output_layout": output_layout,
    "coefficients": {
        "energy": 10,
        "radial_intensity": 10,
        "transverse_intensity": 10,
        "pressure": 0,
        "radial_velocity": 0,
        "transverse_velocity": 0,
        "in_phase_quad": 100,
        "symmetry_quad": 10,
        "in_phase_lin": 0,
        "symmetry_lin": 0,
        "total_gains_lin": 0,
        "total_gains_quad": 0,
    },
    "directional_weights": 1,
    "show_results": False,
    "save_results": True,
    "input_matrix_plots": input_matrix_optimization,
    "cloud_plots": cloud_optimization,
    "results_file_name": "panning502_USAT",
}
optimize(dictionary)


##############################################################


# No optimization ############################################
decoder_matrix = get_input_channels_vbap(cloud_optimization, output_layout).T

speaker_signals = np.dot(input_matrix_optimization, decoder_matrix.T)

show_results = False
save_results = True
save_plot_name = "panning502_direct"
plots_general(
    output_layout,
    speaker_signals,
    cloud_optimization,
    show_results,
    save_results,
    save_plot_name,
)
##############################################################
