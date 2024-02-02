# TRANSCODING MULTICHANNEL 7.1.4 to 5.1.2 (PV)

import numpy as np
from pathlib import Path
from universal_transcoder.auxiliars.get_cloud_points import (
    get_all_sphere_points,
    get_equi_t_design_points,
)
from universal_transcoder.auxiliars.get_input_channels import (
    get_input_channels_vbap,
)
from universal_transcoder.auxiliars.my_coordinates import MyCoordinates
from universal_transcoder.calculations.optimization import optimize


input_layout = MyCoordinates.mult_points(
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

output_layout = MyCoordinates.mult_points(
    np.array(
        [
            (-120, 0, 1),
            (-90, 45, 1),
            (-30, 0, 1),
            (0, 0, 1),
            (30, 0, 1),
            (90, 45, 1),
            (120, 0, 1),
        ]
    )
)

basepath = Path(__file__).resolve().parents[0]
t_design = (
    basepath / "universal_transcoder" / "encoders" / "t-design" / "des.3.56.9.txt"
)
cloud_optimization = get_equi_t_design_points(t_design, False)
cloud_plots = get_all_sphere_points(1, False)
input_matrix_optimization = get_input_channels_vbap(cloud_optimization, input_layout)
input_matrix_plots = get_input_channels_vbap(cloud_plots, input_layout)

dictionary = {
    "input_matrix_optimization": input_matrix_optimization,
    "cloud_optimization": cloud_optimization,
    "output_layout": output_layout,
    "coefficients": {
        "energy": 0,
        "radial_intensity": 0,
        "transverse_intensity": 0,
        "pressure": 10,
        "radial_velocity": 1,
        "transverse_velocity": 1,
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
    "results_file_name": "ex3_714to512_pv",
    "input_matrix_plots": input_matrix_plots,
    "cloud_plots": cloud_plots,
}
print(optimize(dictionary))
