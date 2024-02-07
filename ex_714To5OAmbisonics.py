# TRANSCODING OF 7.1.5 to AMBISONICS 5TH ORDER
from pathlib import Path
import numpy as np
from universal_transcoder.auxiliars.get_cloud_points import (
    get_equi_circumference_points,get_equi_t_design_points,get_all_sphere_points
)
from universal_transcoder.auxiliars.get_input_channels import (
    get_input_channels_vbap, get_input_channels_ambisonics,
)
from universal_transcoder.auxiliars.my_coordinates import MyCoordinates
from universal_transcoder.calculations.optimization import optimize
from universal_transcoder.auxiliars.get_decoder_matrices import get_ambisonics_decoder_matrix
from universal_transcoder.plots_and_logs.all_plots import plots_general

basepath = Path(__file__).resolve().parents[0]

def arrange(cloud):
    ###Add points to elevation plane=0 and remove elevation<0 points because input format is in upper hemisphere
    sph=cloud.sph_deg()
    newSph=[]
    for i in range (sph.shape[0]):
        if(sph[i][1]>=0):
            newSph.append(sph[i])
    zero_elevation=get_equi_circumference_points(10, False)
    zero_elevation_sph=zero_elevation.sph_deg()
    for i in range(zero_elevation_sph.shape[0]):
        newSph.append(zero_elevation_sph[i])
    output_layout=MyCoordinates.mult_points(np.array(newSph))
    return output_layout

def arrange2(cloud):
    ###Just remove elevation<0 points because input format is in upper hemisphere
    sph=cloud.sph_deg()
    newSph=[]
    for i in range (sph.shape[0]):
        if(sph[i][1]>=0):
            newSph.append(sph[i])
    output_layout=MyCoordinates.mult_points(np.array(newSph))
    return output_layout

# USAD #######################################################

# Input Multichannel 7.0.4
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

# Cloud of points to be encoded in input format (7.1.4)
t_design_input = (
    basepath / "universal_transcoder" / "encoders" / "t-design" / "des.3.56.9.txt"
)
cloud_optimization = get_equi_t_design_points(t_design_input, False)
cloud_optimization=arrange(cloud_optimization)

# Input matrix for optimization
input_matrix_optimization = get_input_channels_vbap(cloud_optimization, input_layout)

# Set of virtual output speakers
t_design_output = (
    basepath / "universal_transcoder" / "encoders" / "t-design" / "des.3.40.8.txt"
)
output_layout = get_equi_t_design_points(t_design_output, False)
output_layout=arrange(output_layout)

# Cloud of points to be encoded in input format (7.1.4) for plotting
cloud_plots = get_all_sphere_points(1, False)
cloud_plots=arrange2(cloud_plots)

# Input matrix for plotting
input_matrix_plots = get_input_channels_vbap(cloud_plots, input_layout)

# Transcoding to Ambisonics 5th order. Generate fixed Dspk
order=5
Dspk=get_ambisonics_decoder_matrix(order,output_layout, "pseudo")


dictionary = {
    "input_matrix_optimization": input_matrix_optimization,
    "cloud_optimization": cloud_optimization,
    "output_layout": output_layout, 
    "Dspk":Dspk,
    "ambisonics_encoding_order":5,
    "coefficients": {
        "energy": 0,
        "radial_intensity": 1,
        "transverse_intensity": 1,
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
    "show_results": False,
    "save_results": True,
    "results_file_name": "704transcoding5OA_USAT",
    "input_matrix_plots": input_matrix_plots,
    "cloud_plots": cloud_plots,
}
optimize(dictionary)
#######################################################


# No optimization #####################################
input_matrix=get_input_channels_vbap(cloud_plots, input_layout)

transcoding_matrix= get_input_channels_ambisonics(input_layout, 5)

decoder_matrix=np.dot(Dspk, transcoding_matrix.T)

speaker_signals = np.dot(input_matrix, decoder_matrix.T)

show_results = False
save_results = True
save_plot_name = "704transcoding5OA_direct"
plots_general(output_layout, speaker_signals, cloud_plots, show_results, save_results, "704transcoding5OA_direct")
#######################################################


