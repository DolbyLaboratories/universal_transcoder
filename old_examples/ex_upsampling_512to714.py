# TRANSCODING MULTICHANNEL 5.1.2 to 7.1.4 (EI)

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

show_results = False
save_results = True

## Upsampling with VBAP (channel remapping)
transcoding_matrix_vbap = np.zeros((11, 7))  # Output x input
for ci in range(7):
    vbap_gains = vbap_3D_encoder(input_layout[ci], output_layout)
    transcoding_matrix_vbap[:, ci] = vbap_gains / np.sqrt(np.sum(vbap_gains**2))


## Upsampling with USAT

# Clouds of points
basepath = Path(__file__).resolve().parents[0]
t_design = (
    basepath / "universal_transcoder" / "encoders" / "t-design" / "des.3.56.9.txt"
)
cloud_optimization_list = [
#get_equi_t_design_points(t_design, plot_show=False),
    get_equi_circumference_points(20, plot_show=False),
    input_layout,
    output_layout,
]
cloud_optimization, weights = mix_clouds_of_points(
    cloud_optimization_list,
    list_of_weights=[ 0.4, 0.3, 0.15, 0.15],
    discard_lower_hemisphere=True,
)
cloud_plots = get_all_sphere_points(10, plot_show=False).discard_lower_hemisphere()

# Getting gain matrix
input_matrix_optimization = get_input_channels_vbap(cloud_optimization, input_layout)
input_matrix_plots = get_input_channels_vbap(cloud_plots, input_layout)

optimization_dict = {
    "input_matrix_optimization": input_matrix_optimization,
    "cloud_optimization": cloud_optimization,
    "output_layout": output_layout,
    "coefficients": {
        "energy": 1,
        "radial_intensity": 1,
        "transverse_intensity": 1,
        "pressure": 0,
        "radial_velocity": 0,
        "transverse_velocity": 0,
        "in_phase_quad": 100,
        "symmetry_quad": 1,
        "in_phase_lin": 10,
        "symmetry_lin": 0.1,
        "total_gains_lin": 0,
        "total_gains_quad": 0,
        "sparsity_quad": 0.01,  # 1,
        "sparsity_lin": 0.001,  # 1,
    },
    "directional_weights": 1, #weights,
    "show_results": show_results,
    "save_results": save_results,
    "results_file_name": "ex_upsampling_512to714",
    "input_matrix_plots": input_matrix_plots,
    "cloud_plots": cloud_plots,
    "T_initial": transcoding_matrix_vbap + 0.01*np.random.random_sample(transcoding_matrix_vbap.shape),
}
transcoding_matrix = optimize(optimization_dict)

transcoding_matrix_dB = 20 * np.log10(np.abs(transcoding_matrix))
transcoding_matrix_dB[transcoding_matrix_dB < -30] = -np.inf

np.set_printoptions(precision=1, suppress=True)
print("Transcoding matrix in dB (USAT): ")
print(transcoding_matrix_dB)





transcoding_matrix_vbap_dB= 20 * np.log10(np.abs(transcoding_matrix_vbap))
transcoding_matrix_vbap_dB[transcoding_matrix_vbap_dB < -30] = -np.inf
np.set_printoptions(precision=1, suppress=True)
print("\n Transcoding matri in dB (VBAP)")
print(transcoding_matrix_vbap_dB)

# Corresponding plots
input_matrix_plots = get_input_channels_vbap(cloud_plots, input_layout)
speaker_signals = np.dot(input_matrix_plots, transcoding_matrix_vbap.T)
save_plot_name = "ex_upsampling_512to714_vbap"
plots_general(
    output_layout,
    speaker_signals,
    cloud_plots,
    show_results,
    save_results,
    save_plot_name,
)