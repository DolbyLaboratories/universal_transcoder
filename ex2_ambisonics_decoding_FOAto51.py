# FIRST ORDER AMBISONICS DECODING TO 5.1

import os
from pathlib import Path

import numpy as np
from universal_transcoder.auxiliars.get_cloud_points import (
    get_all_sphere_points,
    get_equi_t_design_points,
    get_equi_circumference_points,
)
from universal_transcoder.auxiliars.get_input_channels import (
    get_input_channels_ambisonics,
)
from universal_transcoder.auxiliars.my_coordinates import MyCoordinates
from universal_transcoder.calculations.optimization import optimize


order = 1

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

cloud_optimization = get_equi_circumference_points(36, False)
cloud_plots = get_equi_circumference_points(360, False)
input_matrix_optimization = get_input_channels_ambisonics(cloud_optimization, order)
input_matrix_plots = get_input_channels_ambisonics(cloud_plots, order)

dictionary = {
    "input_matrix_optimization": input_matrix_optimization,
    "cloud_optimization": cloud_optimization,
    "output_layout": layout_50,
    "coefficients": {
        "energy": 5,
        "radial_intensity": 5,
        "transverse_intensity": 1,
        "pressure": 0,
        "radial_velocity": 0,
        "transverse_velocity": 0,
        "in_phase_quad": 10,
        "symmetry_quad": 2,
        "in_phase_lin": 0,
        "symmetry_lin": 0,
        "total_gains_lin": 0,
        "total_gains_quad": 0,
    },
    "directional_weights": 1,
    "show_results": False,
    "results_file_name": "ex2_ambiFOAto50",
    "save_results": True,
    "input_matrix_plots": input_matrix_plots,
    "cloud_plots": cloud_plots,
}

decoding_matrix = optimize(dictionary)
print(decoding_matrix)