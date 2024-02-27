

import numpy as np
from pathlib import Path
from universal_transcoder.auxiliars.get_cloud_points import (
    get_all_sphere_points,
    get_equi_t_design_points,
    get_equi_circumference_points,
    mix_clouds_of_points,
)
from universal_transcoder.auxiliars.get_input_channels import (
    get_input_channels_vbap,
)
from universal_transcoder.auxiliars.my_coordinates import MyCoordinates
from universal_transcoder.calculations.optimization import optimize
from universal_transcoder.encoders.vbap_encoder import vbap_3D_encoder
from universal_transcoder.plots_and_logs.all_plots import plots_general

##  Input and output layouts definition

input_layout = MyCoordinates.mult_points(
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

output_layout = MyCoordinates.mult_points(
    np.array(
        [
            (10, 0, 1),
            (-45, 0, 1),
            (180, 0, 1),
            (0, 80, 1),
        ]
    )
)

show_results = False
save_results = True



## Transcoding usiing vbap
decoding_matrix_vbap = np.zeros((4, 7))
for ci in range(7):
    vbap_gains = vbap_3D_encoder(input_layout[ci], output_layout)
    decoding_matrix_vbap[:, ci] = vbap_gains / np.sqrt(np.sum(vbap_gains**2))


## Decidubg with USAT

# Clouds of points
basepath = Path(__file__).resolve().parents[0]
t_design = (
    basepath / "universal_transcoder" / "encoders" / "t-design" / "des.3.56.9.txt"
)
cloud_optimization_list = [
    # get_equi_t_design_points(t_design, plot_show=False),
    get_equi_circumference_points(20, plot_show=False),
    input_layout,
    output_layout,
]
cloud_optimization, weights = mix_clouds_of_points(
    cloud_optimization_list,
    list_of_weights=[0.5, 0.3, 0.1, 0.1],
    discard_lower_hemisphere=True,
)
cloud_plots = get_all_sphere_points(10, plot_show=False).discard_lower_hemisphere()

# Getting gain matrix
input_matrix_optimization = get_input_channels_vbap(cloud_optimization, input_layout)
input_matrix_plots = get_input_channels_vbap(cloud_plots, input_layout)

T_initial = decoding_matrix_vbap + 0.001 * np.random.random_sample(
    decoding_matrix_vbap.shape
)
optimization_dict = {
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
        "in_phase_quad": 10000,
        "symmetry_quad": 0,
        "in_phase_lin": 0,
        "symmetry_lin": 0.0,
        "total_gains_lin": 0,
        "total_gains_quad": 0,
        "sparsity_quad": 0.01,  # 1,
        "sparsity_lin": 0.001,  # 1,
    },
    "directional_weights": 1,  # weights,
    "show_results": show_results,
    "save_results": save_results,
    "results_file_name": "ex_decoding_301_irregular",
    "input_matrix_plots": input_matrix_plots,
    "cloud_plots": cloud_plots,
    "T_initial": T_initial,
}
transcoding_matrix = optimize(optimization_dict)


np.set_printoptions(precision=2, suppress=True)

print("\nTranscoding matrix (VBAP)")
decoding_matrix_vbap_dB = 20 * np.log10(np.abs(decoding_matrix_vbap))
decoding_matrix_vbap_dB[decoding_matrix_vbap_dB < -30] = -np.inf
print(decoding_matrix_vbap)
print("In dB:")
print(decoding_matrix_vbap_dB)
# Corresponding plots
input_matrix_plots = get_input_channels_vbap(cloud_plots, input_layout)
speaker_signals = np.dot(input_matrix_plots, decoding_matrix_vbap.T)
save_plot_name = "ex_decoding_301irregular_vbap.png"
plots_general(
    output_layout,
    speaker_signals,
    cloud_plots,
    show_results,
    save_results,
    save_plot_name,
)

print("\nTranscoding matrix in dB (USAT): ")
transcoding_matrix_dB = 20 * np.log10(np.abs(transcoding_matrix))
transcoding_matrix_dB[transcoding_matrix_dB < -30] = -np.inf
print(transcoding_matrix)
print("In dB:")
print(transcoding_matrix_dB)
