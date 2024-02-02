# TRANSCODING MULTICHANNEL 7.1.4 to 5.1.2 (EI)

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

output_layout = MyCoordinates.mult_points(
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

basepath = Path(__file__).resolve().parents[0]
t_design = (
    basepath / "universal_transcoder" / "encoders" / "t-design" / "des.3.56.9.txt"
)


cloud_optimization_list = [
    get_equi_t_design_points(t_design, plot_show=False),
    get_equi_circumference_points(
        20,
        plot_show=False,
    ),
    input_layout,
    output_layout,
]
cloud_optimization_list[0] = cloud_optimization_list[0][
    cloud_optimization_list[0].cart()[:, 2] >= 0
]
n = [x.cart().shape[0] for x in cloud_optimization_list]


cloud_plots = get_all_sphere_points(5, plot_show=False)
cloud_plots = cloud_plots[cloud_plots.cart()[:, 2] >= 0]


# Stack
cloud_optimization = MyCoordinates.mult_points_cart(
    np.vstack([co.cart() for co in cloud_optimization_list])
)

# Weights
weights = np.asarray(
    [0.4 / n[0]] * n[0]
    + [0.3 / n[1]] * n[1]
    + [0.15 / n[2]] * n[2]
    + [0.15 / n[3]] * n[3]
)


input_matrix_optimization = get_input_channels_vbap(cloud_optimization, input_layout)
input_matrix_plots = get_input_channels_vbap(cloud_plots, input_layout)

dictionary = {
    "input_matrix_optimization": input_matrix_optimization,
    "cloud_optimization": cloud_optimization,
    "output_layout": output_layout,
    "coefficients": {
        "energy": 5,
        "radial_intensity": 5,
        "transverse_intensity": 1,
        "pressure": 0,
        "radial_velocity": 0,
        "transverse_velocity": 0,
        "in_phase_quad": 1,
        "symmetry_quad": 1,
        "in_phase_lin": 0,
        "symmetry_lin": 0,
        "total_gains_lin": 0,
        "total_gains_quad": 0,
    },
    "directional_weights": 1,
    "show_results": False,
    "save_results": True,
    "results_file_name": "ex3_714to512_ei",
    "input_matrix_plots": input_matrix_plots,
    "cloud_plots": cloud_plots,
}
print(optimize(dictionary))
