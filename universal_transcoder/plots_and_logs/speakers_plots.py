"""
Copyright (c) 2024 Dolby Laboratories, Amaia Sagasti
Copyright (c) 2023 Dolby Laboratories

Redistribution and use in source and binary forms, with or without modification, are permitted 
provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions 
and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or 
promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED 
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A 
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR 
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED 
TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING 
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
POSSIBILITY OF SUCH DAMAGE.
"""

import warnings
from typing import Union

import matplotlib.pyplot as plt
import numpy as np

from universal_transcoder.auxiliars.my_coordinates import MyCoordinates
from universal_transcoder.auxiliars.typing import NpArray
from universal_transcoder.plots_and_logs.common_plots_functions import save_plot

warnings.filterwarnings(
    "ignore", category=RuntimeWarning
)  # deactivate warnings in log10() calculation


def plot_speaker_2D(
    output_layout: MyCoordinates,
    speaker_signals: NpArray,
    cloud: MyCoordinates,
    save_results: bool,
    results_file_name,
):
    """
    Function to plot speaker gains for each speaker of an output layout. 2D plots.

    Args:
        output_layout (MyCoordinates): positions of output speaker layout:
                pyfar.Coordinates (P-speakers)
        speaker_signals (Array): speaker signals resulting from decoding
                to input set of encoded L directions (LxP size)
        cloud(MyCoordinates): set of points sampling the sphere (L)
        save_results (bool): Flag to save plots
        results_file_name(str): Path where to save the plots
    """
    output_layout = output_layout.sph_deg()

    azimuth = cloud.sph_rad()[:, 0]
    elevation = cloud.sph_rad()[:, 1]
    mask_horizon = (elevation < 0.01) * (elevation > -0.01)

    speaker_gains = speaker_signals
    speaker_gains_db = 20 * np.log10(speaker_gains)

    # mask point at the horizon
    azimuth = azimuth[mask_horizon]
    speaker_gains = speaker_gains[mask_horizon, :]
    speaker_gains_db = speaker_gains_db[mask_horizon, :]

    sum_gains = np.sum(speaker_gains, axis=1)
    sum_gains_db = 20 * np.log10(sum_gains)
    lim = np.max(sum_gains) + 0.1
    lim_db = np.max(sum_gains_db) + 0.5
    lim_db = 5

    fig1 = plt.figure(figsize=(17, 9))

    # Plot in XY
    ax2 = fig1.add_subplot(222)
    for i in range(output_layout.shape[0]):
        ax2.plot(azimuth, speaker_gains[:, i], label=("Speaker nº", i))
    ax2.plot(azimuth, sum_gains.T, label=("Total gain"))
    plt.title("Speaker gains")
    plt.ylim(-0.05, lim)

    # Plot in XY in dBs
    ax3 = fig1.add_subplot(224)
    for i in range(output_layout.shape[0]):
        ax3.plot(azimuth, speaker_gains_db[:, i], label=("Speaker nº", i))
    ax3.plot(azimuth, sum_gains_db, label=("Total gain"))
    plt.title("Speaker gains (dB)")
    plt.ylim(-40, lim_db)

    # Add last to close plot
    azimuth = np.hstack((azimuth, azimuth[0]))
    # print("Shapes ", speaker_gains.shape, speaker_gains[0, :].shape)
    speaker_gains = np.vstack((speaker_gains, speaker_gains[0, :]))
    speaker_gains_db = np.vstack((speaker_gains_db, speaker_gains_db[0, :]))
    sum_gains = np.hstack((sum_gains, sum_gains[0]))
    sum_gains_db = np.hstack((sum_gains_db, sum_gains_db[0]))

    # Plot in Polar

    ax = fig1.add_subplot(221, projection="polar")
    ax.legend(bbox_to_anchor=(0.5, -0.55), loc="lower center")
    for i in range(output_layout.shape[0]):
        ax.plot(azimuth, speaker_gains[:, i].T, label=("Speaker nº", i))
    ax.plot(azimuth, sum_gains.T, label=("Total gain"))
    ax.set_theta_zero_location("N")
    plt.title("Speaker gains")
    plt.ylim(0, lim)

    # plt.show()

    # Save plots
    if save_results and (type(results_file_name) == str):
        file_name = "plot_speaker_gains_2D.png"
        save_plot(plt, results_file_name, file_name)


def plot_speaker_3D(
    output_layout: MyCoordinates,
    speaker_signals: NpArray,
    cloud_points: MyCoordinates,
    save_results: bool,
    results_file_name: Union[bool, str] = False,
):
    """
    Function to plot speaker gains for each speaker of an output layout.
    3D projection on 2D plots.

    Args:
        output_layout (MyCoordinates): positions of output speaker layout:
                pyfar.Coordinates (P-speakers)
        speaker_signals (Array): speaker signals resulting from decoding
                to input set of encoded L directions (LxP size)
        cloud(MyCoordinates): set of points sampling the sphere (L)
        save_results (bool): Flag to save plots
        results_file_name(str): Path where to save the plots
    """

    # Plot
    cloud_points_cart = cloud_points.cart()
    X = cloud_points_cart[:, 0]
    Y = cloud_points_cart[:, 1]
    Z = cloud_points_cart[:, 2]
    layout_sph = output_layout.sph_deg()
    number_spk = layout_sph.shape[0]
    fig = plt.figure(figsize=(17, 9))
    for i in range(number_spk):
        ax = fig.add_subplot(1, number_spk, i + 1, projection="3d")
        sc = ax.scatter(X, Y, Z, c=speaker_signals[:, i], cmap="rainbow")
        ax.axis("equal")
        ax.set_xlabel("X axis", fontsize=6)
        ax.set_ylabel("Y axis", fontsize=6)
        ax.set_zlabel("Z axis", fontsize=6)
        ax.tick_params(labelsize=6)
        title = "Speaker position {}"
        plt.title(title.format(layout_sph[i, :]), fontsize=7)

        sc.set_clim(0, 1)
    plt.colorbar(sc)  # To see colorbar scale
    # plt.show()

    # Save plots
    if save_results and (type(results_file_name) == str):
        file_name = "plot_speaker_gains_3D.png"
        save_plot(plt, results_file_name, file_name)
