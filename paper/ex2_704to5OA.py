"""
Copyright (c) 2024 Dolby Laboratories, Amaia Sagasti

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

# TRANSCODING OF 7.1.5 to AMBISONICS 5TH ORDER
from pathlib import Path
import numpy as np
from universal_transcoder.auxiliars.get_cloud_points import (
    get_equi_circumference_points,
    get_equi_t_design_points,
    get_all_sphere_points,
    mix_clouds_of_points,
)
from universal_transcoder.auxiliars.get_input_channels import (
    get_input_channels_vbap,
    get_input_channels_ambisonics,
)
from universal_transcoder.auxiliars.my_coordinates import MyCoordinates
from universal_transcoder.calculations.optimization import optimize
from universal_transcoder.auxiliars.get_decoder_matrices import (
    get_ambisonics_decoder_matrix,
)
from universal_transcoder.plots_and_logs.all_plots import plots_general
from universal_transcoder.plots_and_logs.import_allrad_dec import get_allrad_decoder

basepath = Path(__file__).resolve().parents[0]

# USAT #######################################################

# Input Multichannel 7.0.4
input_layout = MyCoordinates.mult_points(
    np.array(
        [
            (30, 0, 1),  # L
            (-30, 0, 1),  # R
            (0, 0, 1),  # C
            (90, 0, 1),  # Ls
            (-90, 0, 1),  # Rs
            (120, 0, 1),  # Lb
            (-120, 0, 1),  # Rb
            (45, 45, 1),  # Tfl
            (-45, 45, 1),  # Tfr
            (135, 45, 1),  # Tbl
            (-135, 45, 1),  # Tbr
        ]
    )
)

# Cloud of points to be encoded in input format (7.1.4)
t_design_input = (
    basepath / "universal_transcoder" / "encoders" / "t-design" / "des.3.56.9.txt"
)
cloud_optimization_list = [
    get_equi_t_design_points(t_design_input, False),
    get_equi_circumference_points(15, False),
    input_layout,
]
cloud_optimization, weights = mix_clouds_of_points(
    cloud_optimization_list,
    list_of_weights=[0.6, 0.3, 0.1],
    discard_lower_hemisphere=True,
)

# Input matrix for optimization
input_matrix_optimization = get_input_channels_vbap(cloud_optimization, input_layout)

# Set of virtual output speakers
t_design_output = (
    basepath / "universal_transcoder" / "encoders" / "t-design" / "des.3.60.10.txt"
)
output_layout, _ = mix_clouds_of_points(
    [
        get_equi_t_design_points(t_design_output, False),
        get_equi_circumference_points(36, False),
    ],
    list_of_weights=[1, 1],
    discard_lower_hemisphere=True,
)

# Cloud of points to be encoded in input format (7.1.4) for plotting
cloud_plots = get_all_sphere_points(1, False).discard_lower_hemisphere()

# Input matrix for plotting
input_matrix_plots = get_input_channels_vbap(cloud_plots, input_layout)

# Transcoding to Ambisonics 5th order. Generate fixed Dspk
order = 5
Dspk = get_ambisonics_decoder_matrix(order, output_layout, "pseudo")


dictionary = {
    "input_matrix_optimization": input_matrix_optimization,
    "cloud_optimization": cloud_optimization,
    "output_layout": output_layout,
    "Dspk": Dspk,
    "ambisonics_encoding_order": 5,
    "coefficients": {
        "energy": 0,
        "radial_intensity": 0.2,
        "transverse_intensity": 0.1,
        "pressure": 5,
        "radial_velocity": 2,
        "transverse_velocity": 1,
        "in_phase_quad": 0,
        "symmetry_quad": 0,
        "in_phase_lin": 0,
        "symmetry_lin": 0,
        "total_gains_lin": 0,
        "total_gains_quad": 0,
    },
    "directional_weights": weights,
    "show_results": True,
    "save_results": False,
    "results_file_name": "ex2_704to5OA_USAT",
    "input_matrix_plots": input_matrix_plots,
    "cloud_plots": cloud_plots,
}
optimize(dictionary)
#######################################################


# No optimization #####################################
input_matrix = get_input_channels_vbap(cloud_plots, input_layout)

transcoding_matrix = get_input_channels_ambisonics(input_layout, 5)

decoder_matrix = np.dot(Dspk, transcoding_matrix.T)

speaker_signals = np.dot(input_matrix, decoder_matrix.T)

show_results = True
save_results = False
save_plot_name = "ex2_704to5OA_direct"
plots_general(
    output_layout,
    speaker_signals,
    cloud_plots,
    show_results,
    save_results,
    save_plot_name,
)
#######################################################
