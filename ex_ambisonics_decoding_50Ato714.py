# FIFTH ORDER AMBISONICS DECODING TO 7.1.4.

import os
from pathlib import Path

import numpy as np
from universal_transcoder.auxiliars.get_cloud_points import (
    get_all_sphere_points,
    get_equi_t_design_points,
)
from universal_transcoder.auxiliars.get_input_channels import (
    get_input_channels_ambisonics,
)
from universal_transcoder.auxiliars.my_coordinates import MyCoordinates
from universal_transcoder.calculations.optimization import optimize
from universal_transcoder.plots_and_logs.all_plots import plots_general
from universal_transcoder.plots_and_logs.import_allrad_dec import get_allrad_decoder

order = 5
layout_704 = MyCoordinates.mult_points(
    np.array(
        [
            (30, 0, 1),
            (-30, 0, 1),
            (0, 0, 1),
            (90, 0, 1),
            (-90, 0, 1),
            (120, 0, 1),
            (-120, 0, 1),
            (45, 45, 1),
            (-45, 45, 1),
            (135, 45, 1),
            (-135, 45, 1),
        ]
    )
)

basepath = Path(__file__).resolve().parents[0]
t_design = (
    basepath / "universal_transcoder" / "encoders" / "t-design" / "des.3.56.9.txt"
)
cloud_optimization = get_equi_t_design_points(t_design, False)
cloud_plots = get_all_sphere_points(1, False)
input_matrix_optimization = get_input_channels_ambisonics(cloud_optimization, order)
input_matrix_plots = get_input_channels_ambisonics(cloud_plots, order)

dictionary = {
    "input_matrix_optimization": input_matrix_optimization,
    "cloud_optimization": cloud_optimization,
    "output_layout": layout_704,
    "coefficients": {
        "energy": 5,
        "radial_intensity": 2,
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
    "show_results": True,
    "results_file_name": "ex_5OAto704",
    "save_results": True,
    "input_matrix_plots": input_matrix_plots,
    "cloud_plots": cloud_plots,
}

decoding_matrix = optimize(dictionary)
print(decoding_matrix)


### AllRad

file_name = "allrad_704_5OA_basic_decoder.json"
order = 5

# Import AllRad file (N3D and ACN)
decoding_matrix = get_allrad_decoder(
    "allrad_decoders/" + file_name,
    type="maxre",
    order=order,
    convention="sn3d",
    normalize_energy=True,
    layout=layout_704,
)
output_layout = layout_704
# Input channels
input_matrix = get_input_channels_ambisonics(cloud_plots, order)

# Speaker matrix
speaker_matrix = np.dot(input_matrix, decoding_matrix.T)

# Call plots and save results
show_results = True
save_results = True
save_plot_name = "paper_case2_ambi5OAto704_allrad_maxre"
plots_general(
    output_layout,
    speaker_matrix,
    cloud_plots,
    show_results,
    save_results,
    save_plot_name,
)
