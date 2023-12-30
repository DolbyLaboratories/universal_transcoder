"""
Copyright 2023 Dolby Laboratories

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

import matplotlib.pyplot as plt
import numpy as np
from universal_transcoder.auxiliars.my_coordinates import MyCoordinates
from universal_transcoder.plots_and_logs.e_i_plots import plot_ei_2D, plot_ei_3D
from universal_transcoder.plots_and_logs.p_v_plots import plot_pv_2D, plot_pv_3D
from universal_transcoder.plots_and_logs.speakers_plots import (
    plot_speaker_2D,
)


def plots_general(
    output_layout: MyCoordinates,
    speaker_signals: np,
    cloud: MyCoordinates,
    show_results: bool,
    save_results: bool,
    save_plot_name,
    normalize: str = None,
):
    """
    Function group all plots and call them at once

    Args:
        output_layout (MyCoordinates): positions of output speaker layout:
                pyfar.Coordinates (N-speakers)
        speaker_signals (numpy Array): speaker signals resulting from decoding 
                to input set of encoded L directions (LxN size)
        cloud(MyCoordinates): set of points sampling the sphere (L)
        show_results (bool): Flag to show plots
        save_results (bool): Flag to save plots
        results_file_name(str): Path where to save the plots
    """
    plot_ei_2D(
        output_layout,
        speaker_signals,
        cloud,
        save_results,
        save_plot_name,
        normalize=normalize,
    )
    plot_pv_2D(
        output_layout,
        speaker_signals,
        cloud,
        save_results,
        save_plot_name,
        normalize=normalize,
    )
    plot_speaker_2D(
        output_layout,
        speaker_signals,
        cloud,
        save_results,
        save_plot_name,
        normalize=normalize,
    )
    plot_ei_3D(
        output_layout,
        speaker_signals,
        cloud,
        save_results,
        save_plot_name,
        normalize=normalize,
    )
    plot_pv_3D(
        output_layout,
        speaker_signals,
        cloud,
        save_results,
        save_plot_name,
        normalize=normalize,
    )

    if show_results:
        plt.show()
