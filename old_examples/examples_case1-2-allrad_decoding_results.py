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
from universal_transcoder.plots_and_logs.import_allrad_dec import get_allrad_decoder
from universal_transcoder.plots_and_logs.all_plots import plots_general
from universal_transcoder.auxiliars.get_cloud_points import (
    get_equi_circumference_points,
    get_all_sphere_points,
)
from universal_transcoder.auxiliars.get_input_channels import (
    get_input_channels_ambisonics,
)
from universal_transcoder.plots_and_logs.common_plots_functions import normalize_S
from universal_transcoder.auxiliars.my_coordinates import MyCoordinates


file_name = "allrad_704_5OA_basic_decoder.json"
order = 5




# Cloud
# cloud = get_equi_circumference_points(36, False)
cloud = get_all_sphere_points(5, False)
# Output layout
layout_50 = MyCoordinates.mult_points(
    np.array(
        [
            (-120, 0, 1),
            (-30, 0, 1),
            (0, 0, 1),
            (30, 0, 1),
            (120, 0, 1),
        ]
    )
)
layout_704 = MyCoordinates.mult_points(
    np.array(
        [
            (-135, 45, 1),
            (-120, 0, 1),
            (-90, 0, 1),
            (-45, 45, 1),
            (-30, 0, 1),
            (0, 0, 1),
            (30, 0, 1),
            (45, 45, 1),
            (90, 0, 1),
            (120, 0, 1),
            (135, 45, 1),
        ]
    )
)


# Import AllRad file (N3D and ACN)
decoding_matrix = get_allrad_decoder(
    "allrad_decoders/" + file_name, "maxre", order, "sn3d", normalize_energy=True, layout=layout_704
)
output_layout = layout_704
# Input channels
input_matrix = get_input_channels_ambisonics(cloud, order)

# Speaker matrix
speaker_matrix = np.dot(input_matrix, decoding_matrix.T)


# Call plots and save results
show_results = True
save_results = True
save_plot_name = "paper_case2_ambi5OAto704_allrad_maxre"
plots_general(
    output_layout,
    speaker_matrix,
    cloud,
    show_results,
    save_results,
    save_plot_name,
)
