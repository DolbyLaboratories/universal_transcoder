import numpy as np

from universal_transcoder.auxiliars.get_input_channels import (
    get_input_channels_micro_cardioid,
    get_input_channels_micro_omni,
)
from universal_transcoder.auxiliars.get_cloud_points import (
    get_equi_circumference_points,
)
from universal_transcoder.auxiliars.my_coordinates import MyCoordinates
from universal_transcoder.calculations.optimization import optimize

# MIC EXAMPLE - dictionary - perfect

mic_orientation = np.array(
    [
        (-90, 0),
        (0, 0),
        (90, 0),
        (180, 0),
    ]
)

# microphones have to be coincident
mic_position = MyCoordinates.mult_points(
    np.array(
        [
            (-90, 0, 0),
            (0, 0, 0),
            (90, 0, 0),
            (180, 0, 0),
        ]
    )
)

# Output layout
output_layout = MyCoordinates.mult_points(
    np.array(
        [
            (-90.1, 0, 1),
            (0.1, 0, 1),
            (90.1, 0, 1),
            (179.9, 0, 1),
        ]
    )
)

# Cloud
cloud = get_equi_circumference_points(31, False)
cloud_plots = get_equi_circumference_points(360, False)

# frequency
freq = 100
in_chans = get_input_channels_micro_cardioid(
    cloud, mic_orientation, mic_position, freq
)
in_chans = np.array(np.abs(in_chans))

input_matrix_plots = get_input_channels_micro_cardioid(
    cloud_plots, mic_orientation, mic_position, freq
)
input_matrix_plots = np.array(np.abs(input_matrix_plots))


dictionary = {
    "input_format": "microphone",
    "input_matrix_optimization": in_chans,
    "cloud_optimization": cloud,
    "frequency": freq,
    "transcoding": True,
    "ambisonics_encoding_order": 1,
    "output_layout": output_layout,
    "coefficients": {
        "energy": 0,
        "radial_intensity": 0,
        "transverse_intensity": 0,
        "pressure": 100,
        "radial_velocity": 20,
        "transverse_velocity": 2,
        "in_phase_quad": 0,
        "symmetry_quad": 0,
        "total_gains_quad": 0,
        "in_phase_lin": 0,
        "symmetry_lin": 1,
        "total_gains_lin": 0,
    },
    "input_matrix_plots": input_matrix_plots,
    "cloud_plots": cloud_plots,
    "directional_weights": 1,
    "show_results": False,
    "results_file_name": "microphone_to_amb_over",
    "save_results": True,
    "channels_labels": [str(idx) for idx in range(len(mic_orientation))],
}
transcoding_matrices = optimize(dictionary)

