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
basepath = Path(__file__).resolve().parents[0]

# USAT #######################################################

# Cloud of points to be encoded in input format (5OA)
t_design = (
    basepath / "universal_transcoder" / "encoders" / "t-design" / "des.3.56.9.txt"
)
cloud_optimization = get_equi_t_design_points(t_design, False)

#Input Ambisonics 5th Order
order = 5
input_matrix_optimization = get_input_channels_ambisonics(cloud_optimization, order)

# Output Layout of speakers 7.1.4 (we are decoding, no virtual speakers)
output_layout = MyCoordinates.mult_points(
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

# Cloud of points to be encoded in input format (5OA) for plotting
cloud_plots = get_all_sphere_points(1, False)

# Input matrix for plotting
input_matrix_plots = get_input_channels_ambisonics(cloud_plots, order)


dictionary = {
    "input_matrix_optimization": input_matrix_optimization,
    "cloud_optimization": cloud_optimization,
    "output_layout": output_layout,
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
    "results_file_name": "5OAdecoding714",
    "save_results": True,
    "input_matrix_plots": input_matrix_plots,
    "cloud_plots": cloud_plots,
}

optimize(dictionary)
#######################################################

# No optimization #####################################
### AllRad

file_name = "allrad_704_5OA_basic_decoder.json"
order = 5

# Import AllRad file (N3D and ACN)
decoding_matrix = get_allrad_decoder(
    "allrad_decoders/" + file_name,
    type="basic", # basic / maxre / inphase
    order=order,
    convention="sn3d",
    normalize_energy=True,
    layout=output_layout,
)

# Input channels
input_matrix = get_input_channels_ambisonics(cloud_plots, order)

# Speaker matrix
speaker_matrix = np.dot(input_matrix, decoding_matrix.T)

# Call plots and save results
show_results = True
save_results = True
save_plot_name = "5OAdecoding714_allrad_basic" # basic / maxre / inphase
plots_general(
    output_layout,
    speaker_matrix,
    cloud_plots,
    show_results,
    save_results,
    save_plot_name,
)
#######################################################
