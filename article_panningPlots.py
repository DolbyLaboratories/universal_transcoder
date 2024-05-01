import warnings
from pathlib import Path

import numpy as np
import sys
import os

from universal_transcoder.auxiliars.get_cloud_points import (
    get_equi_circumference_points,
)
from universal_transcoder.auxiliars.get_input_channels import (
    get_input_channels_vbap,
    get_input_channels_ambisonics,
)
from universal_transcoder.encoders.vbap_encoder import vbap_3D_encoder
from universal_transcoder.plots_and_logs.import_allrad_dec import get_allrad_decoder
from universal_transcoder.auxiliars.my_coordinates import MyCoordinates

import matplotlib.pyplot as plt
from matplotlib import rc

rc(
    "font",
    **{
        "family": "serif",
        "serif": ["Times"],
        "size": 9,  # Equivalent to \small in LaTeX
    }
)
plt.rc("axes", **{"labelsize": 9, "titlesize": 9})
rc("text", usetex=True)

from universal_transcoder.auxiliars.my_coordinates import MyCoordinates
from universal_transcoder.plots_and_logs.common_plots_functions import save_plot

warnings.filterwarnings(
    "ignore", category=RuntimeWarning
)  # deactivate warnings in log10() calculation


def panning_plots_2d(
    input_matrix,
    decoder_matrix,
    output_layout: MyCoordinates,
    cloud: MyCoordinates,
    path,
    title,
    legend_tags=None,
):

    # Prepare data
    output_layout = output_layout.sph_deg()
    azimuth = cloud.sph_deg()[:, 0]
    elevation = cloud.sph_rad()[:, 1]
    speaker_gains = np.dot(input_matrix, decoder_matrix.T)
    speaker_gains_db = 20 * np.log10(speaker_gains)

    # Remove information of cloud points out of elevation = 0 plane
    mask_horizon = (elevation < 0.01) * (elevation > -0.01)
    azimuth = azimuth[mask_horizon]
    speaker_gains = speaker_gains[mask_horizon, :]
    speaker_gains_db = speaker_gains_db[mask_horizon, :]

    # Remove information of speakers out of elevation = 0 plane
    mask_condition = np.abs(output_layout[:, 1]) < 0.01
    output_layout = output_layout[mask_condition]
    speaker_gains = speaker_gains[:, mask_condition]
    speaker_gains_db = speaker_gains_db[:, mask_condition]

    # Vertical lines - position of output speakers
    vertical_lines = np.sort(output_layout[:, 0])

    # Limits
    lim_min = -0.15
    lim_min_db = -40  # np.max(speaker_gains_db) - 0.5
    lim_max = 1.3
    lim_max_db = 1  # np.max(speaker_gains_db) + 0.5

    # Size
    mm = 1 / 25.4  # mm in inches
    width = 76 * mm  # Replace 76 by 159 for 2-column plots
    aspect_ratio = 0.6  # Replace by whatever apprpriate
    fig, ax = plt.subplots(figsize=(width, aspect_ratio * width))

    # Legends posibilities
    if legend_tags is None:
        legend_tags = [
            "L",
            "R",
            "C",
            "Ls",
            "Rs",
            "Lb",
            "Rb",
            "Tfl",
            "Tfr",
            "Tbl",
            "Tbr",
        ]

    # Clear
    plt.clf()

    # Plot in Cartesian - Linear
    ax = fig.add_subplot(111)
    for i in range(output_layout.shape[0]):
        ax.plot(azimuth, speaker_gains[:, i], label=(legend_tags[i]))
    # Legend
    ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
    # Set specific tick marks
    xtick_positions = [-180, -90, 0, 90, 180]
    plt.xticks(xtick_positions)
    ytick_positions = [-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1, 1.2]
    plt.yticks(ytick_positions)
    # Set vertical lines
    for line in vertical_lines:
        plt.axvline(x=line, color=(0.7, 0.7, 0.7), linestyle="--", linewidth=1)
    # Axis limits
    plt.xlim(180, -180)
    plt.ylim(lim_min, lim_max)
    # Axis titles
    plt.xlabel("azimuth (deg)")
    plt.ylabel("gain (linear)") # Set title for the x-axis
    # Plot title
    plt.title(title)
    plt.grid()
    # Adjust the size of the graph within the figure
    left, bottom, width, height = 0.12, 0.2, 0.6, 0.6
    ax.set_position([left, bottom, width, height])
    # Save plots
    if type(path) is str:

        name = "speaker_gains-linear-" + Path(path).stem  + ".png"
        save_plot(plt, path, name)
        name = "speaker_gains-linear-" + Path(path).stem  + ".pdf"
        save_plot(plt, path, name)
    # Clear
    plt.clf()

    # Plot in Cartesian - dB
    ax = fig.add_subplot(111)
    for i in range(output_layout.shape[0]):
        ax.plot(azimuth, speaker_gains_db[:, i], label=(legend_tags[i]))
    # Legend
    ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
    # Set specific tick marks
    xtick_positions = [-180, -90, 0, 90, 180]
    plt.xticks(xtick_positions)
    ytick_positions = [-40, -20, -10, -5, 0]
    plt.yticks(ytick_positions)
    # Set vertical lines
    for line in vertical_lines:
        plt.axvline(x=line, color=(0.7, 0.7, 0.7), linestyle="--", linewidth=1)
    # Axis limits
    plt.xlim(180, -180)
    plt.ylim(lim_min_db, lim_max_db)
    # Axis titles
    plt.xlabel("azimuth (deg)")
    plt.ylabel("gain (dB)")
    # Title
    plt.title(title)
    # Turn grid on
    plt.grid()
    # Adjust the size of the graph within the figure
    left, bottom, width, height = 0.12, 0.2, 0.6, 0.6
    ax.set_position([left, bottom, width, height])
    # Save plots
    if type(path) is str:
        name = "speaker_gains-dB-" + Path(path).stem  + ".png"
        save_plot(plt, path, name)
        name = "speaker_gains-dB-" + Path(path).stem  + ".pdf"
        save_plot(plt, path, name)
    # Clear
    plt.clf()


# 5.1 panning - Optimal decoder########################################
# Cloud of points to be encoded in input format (one-to-one panning)
L = 72
cloud = get_equi_circumference_points(L, False)
# Input matrix
input_matrix = np.identity(L)
# Set of output speakers
output_layout = MyCoordinates.mult_points(
    np.array(
        [
            (30, 0, 1),  # L
            (-30, 0, 1),  # R
            (0, 0, 1),  # C
            (120, 0, 1),  # Ls
            (-120, 0, 1),  # Rs
        ]
    )
)
## USAT
# Decoding matrix
folder_name = "panning51_USAT/"
full_path = "saved_results/" + folder_name
decoder_matrix = np.load(full_path + "panning51_USAT.npy")
title = "USAT"
panning_plots_2d(input_matrix, decoder_matrix, output_layout, cloud, folder_name, title)
## VBAP
folder_name = "panning51_direct/"
title = "Tangent law / VBAP"
decoder_matrix = get_input_channels_vbap(cloud, output_layout).T
panning_plots_2d(input_matrix, decoder_matrix, output_layout, cloud, folder_name, title)
## VBIPP
folder_name = "panning51_vbip/"
title = "VBIP"
decoder_matrix = get_input_channels_vbap(cloud, output_layout, vbip=True).T
panning_plots_2d(input_matrix, decoder_matrix, output_layout, cloud, folder_name, title)

############################################################################

# 5OA Decoding to 7.0.4 ####################################################
# Cloud of points to be encoded in input format (5OA) for plotting
cloud = get_equi_circumference_points(360, False)
# Input matrix
order = 5
input_matrix = get_input_channels_ambisonics(cloud, order)
# Set of output speakers
output_layout = MyCoordinates.mult_points(
    np.array(
        [
            (30, 0, 1),  # L
            (-30, 0, 1),  # R
            (0, 0, 1),  # C
            (90, 0, 1),  # Ls
            (-90, 0, 1),  # Rs
            (120, 0, 1),  # Lb
            (-120, 0, 1),  # Rb
            (45, 45, 1),  # Tfl
            (-45, 45, 1),  # Tfr
            (135, 45, 1),  # Tbl
            (-135, 45, 1),  # Tbr
        ]
    )
)
## USAT
# Decoding matrix
folder_name = "5OAdecoding714_USAT/"
full_path = "saved_results/" + folder_name
decoder_matrix = np.load(full_path + "5OAdecoding714_USAT.npy")
title = "USAT"
panning_plots_2d(input_matrix, decoder_matrix, output_layout, cloud, folder_name, title)
## AllRad
folder_name = "5OAdecoding714_allrad_maxre/"
title = "AllRad"
file_name = "704ordered_decoder.json"
decoder_matrix = get_allrad_decoder(
    "allrad_decoders/" + file_name,
    type="maxre",  # basic / maxre / inphase
    order=order,
    convention="sn3d",
    normalize_energy=True,
    layout=output_layout,
)
panning_plots_2d(input_matrix, decoder_matrix, output_layout, cloud, folder_name, title)

############################################################################


############################################################################

# 5.0.1 DECODING TO 3.0.1 ####################################################
# Cloud of points to be encoded in input format (5OA) for plotting
cloud = get_equi_circumference_points(360, False)
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
# Input matrix
input_matrix = get_input_channels_vbap(cloud, input_layout)
# Set of output speakers
## USAT
# Decoding matrix
folder_name = "ex_decoding_301_irregular/"
full_path = "saved_results/" + folder_name
decoder_matrix = np.load(full_path + "ex_decoding_301_irregular.npy")
title = "USAT"
panning_plots_2d(
    input_matrix,
    decoder_matrix,
    output_layout,
    cloud,
    folder_name,
    title,
    legend_tags=["L", "R", "S", "T"],
)
## remapping
folder_name = "ex_decoding_301_irregular_vbap"
title = "Channel remapping (VBAP)"
decoding_matrix_vbap = np.zeros((4, 7))
for ci in range(7):
    vbap_gains = vbap_3D_encoder(input_layout[ci], output_layout)
    decoding_matrix_vbap[:, ci] = vbap_gains / np.sqrt(np.sum(vbap_gains**2))

panning_plots_2d(
    input_matrix,
    decoding_matrix_vbap,
    output_layout,
    cloud,
    folder_name,
    title,
    legend_tags=["L", "R", "S", "T"],
)



############################################################################

