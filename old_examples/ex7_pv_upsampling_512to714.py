# TRANSCODING MULTICHANNEL 5.1.2 to 7.1.4(PV)

import numpy as np
from pathlib import Path
from universal_transcoder.auxiliars.get_cloud_points import (
    get_all_sphere_points,
    get_equi_t_design_points,
    get_equi_circumference_points,
)
from universal_transcoder.auxiliars.get_input_channels import (
    get_input_channels_vbap,
)
from universal_transcoder.auxiliars.my_coordinates import MyCoordinates
from universal_transcoder.calculations.optimization import optimize

input_layout = MyCoordinates.mult_points(
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

output_layout = MyCoordinates.mult_points(
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


basepath = Path(__file__).resolve().parents[0]
t_design = (
    basepath / "universal_transcoder" / "encoders" / "t-design" / "des.3.56.9.txt"
)
cloud_optimization = get_equi_t_design_points(t_design, False)
cloud_plots = get_all_sphere_points(1, False)
input_matrix_optimization = get_input_channels_vbap(cloud_optimization, input_layout)
input_matrix_plots = get_input_channels_vbap(cloud_plots, input_layout)

"""
#AUX
input_layout = MyCoordinates.mult_points(
    np.array(
        [
            (-120, 0, 1),

            (0, 0, 1),

            (120, 0, 1),
        ]
    )
)
output_layout = MyCoordinates.mult_points(
    np.array(
        [

            (-135,0,1),
            (-45,0,1),
            (45, 0, 1),
            (135,0,1),
        ]
    )
)
cloud_optimization = get_equi_circumference_points(36)
cloud_plots = get_equi_circumference_points(360)
input_matrix_optimization = get_input_channels_vbap(cloud_optimization, input_layout)
input_matrix_plots = get_input_channels_vbap(cloud_plots, input_layout)"""

dictionary = {
    "input_matrix_optimization": input_matrix_optimization,
    "cloud_optimization": cloud_optimization,
    "output_layout": output_layout,
    "coefficients": {
        "energy": 1,  # 1,
        "radial_intensity": 1,  # 1,
        "transverse_intensity": 1,  # 1,
        "pressure": 0,
        "radial_velocity": 0,
        "transverse_velocity": 0,
        "in_phase_quad": 0,
        "symmetry_quad": 0,
        "in_phase_lin": 0,
        "symmetry_lin": 0,
        "total_gains_lin": 0,
        "total_gains_quad": 0,
    },
    "directional_weights": 1,
    "show_results": True,
    "save_results": False,
    "results_file_name": "ex7_512to714_pv",  # "aux_30to40_pv"
    "input_matrix_plots": input_matrix_plots,
    "cloud_plots": cloud_plots,
}
print(optimize(dictionary))
