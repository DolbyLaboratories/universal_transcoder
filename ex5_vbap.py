
import numpy as np
from universal_transcoder.auxiliars.get_cloud_points import (
    get_equi_circumference_points,
)
from universal_transcoder.auxiliars.my_coordinates import MyCoordinates
from universal_transcoder.plots_and_logs.all_plots import plots_general
from universal_transcoder.auxiliars.get_input_channels import (
    get_input_channels_vbap,
)

# Encoding multichannel without vbap
output_layout = MyCoordinates.mult_points(
    np.array(
        [
            (-120, 0, 1),
            (-30,0,1),
            (0, 0, 1),
            (30,0,1),
            (120, 0, 1),
        ]
    )
)

n = 72
# Data for plots
cloud = get_equi_circumference_points(n, False)
input_matrix = np.identity(n)

decoding_matrix = get_input_channels_vbap(cloud, output_layout).T
speaker_signals = np.dot(input_matrix, decoding_matrix.T)
# Call plots and save results
show_results = True
save_results = True
save_plot_name = "ex5_VBAP51"
plots_general(
    output_layout,
    speaker_signals,
    cloud,
    show_results,
    save_results,
    save_plot_name,
)
