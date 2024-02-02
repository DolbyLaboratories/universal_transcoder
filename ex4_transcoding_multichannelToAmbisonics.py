# TRANSCODING MULTICHANNEL 5.1. to First order Ambisonics

import numpy as np
from universal_transcoder.auxiliars.get_cloud_points import (
    get_equi_circumference_points,
    get_equi_t_design_points,
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


input_layout = MyCoordinates.mult_points(
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
output_layout = MyCoordinates.mult_points(
    np.array(
        [
            (-180, 0, 1),
            (-90, 0, 1),
            (0, -90, 1),
            (0, 0, 1),
            (0, 90, 1),
            (90, 0, 1),
        ]
    )
)

# output_layout = get_equi_circumference_points(20, False)
cloud_optimization = get_equi_circumference_points(36, False)
input_matrix_optimization = get_input_channels_vbap(cloud_optimization, input_layout)
cloud_plots = get_equi_circumference_points(360, False)
input_matrix_plots = get_input_channels_vbap(cloud_plots, input_layout)

# Transcoding to Ambisonics First order. Generate Dspk
order = 1
Dspk = get_ambisonics_decoder_matrix(order, output_layout, "pseudo")


dictionary = {
    "input_matrix_optimization": input_matrix_optimization,
    "cloud_optimization": cloud_optimization,
    "output_layout": output_layout,
    "Dspk": Dspk,
    "ambisonics_encoding_order": 1,
    "coefficients": {
        "energy": 0,
        "radial_intensity": 0,
        "transverse_intensity": 0,
        "pressure": 100,
        "radial_velocity": 0.1,
        "transverse_velocity": 0.1,
        "in_phase_quad": 0,
        "symmetry_quad": 0,
        "in_phase_lin": 0,
        "symmetry_lin": 0,
        "total_gains_lin": 0,
        "total_gains_quad": 0,
    },
    "directional_weights": 1,
    "show_results": True,
    "save_results": True,
    "results_file_name": "ex4_2D_FOAencoding_pv",
    "input_matrix_plots": input_matrix_plots,
    "cloud_plots": cloud_plots,
}
print(optimize(dictionary))
