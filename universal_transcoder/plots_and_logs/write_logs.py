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

import copy
import json
import os
from typing import Dict, Any

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap

from universal_transcoder.auxiliars.my_coordinates import MyCoordinates
from universal_transcoder.auxiliars.typing import Array
from universal_transcoder.calculations.cost_function import State


def generate_json(data: dict, route_name: str):
    """
    Function to generate and save json of input dictionary

    Args:
        data (dict): information to be saved
        route_name (str): full path name
    """

    data_dict = data.copy()
    # Input channels - save separately real and imaginary parts
    data_dict["input_matrix_optimization"] = data_dict[
        "input_matrix_optimization"
    ].tolist()

    # Adapt cloud format
    data_dict["cloud_optimization"] = data_dict["cloud_optimization"].sph_deg().tolist()

    # Adapt output layout format
    aux = data_dict["output_layout"]
    data_dict["output_layout"] = aux.sph_deg().tolist()

    # Plots input matrix and cloud
    if "input_matrix_plots" in data_dict:
        # Input channels - save separately real and imaginary parts
        data_dict["input_matrix_plots"] = data_dict["input_matrix_plots"].tolist()
        # Adapt cloud format
        data_dict["cloud_plots"] = data_dict["cloud_plots"].sph_deg().tolist()

    if "Dspk" in data_dict:
        data_dict["Dspk"] = data_dict["Dspk"].tolist()

    if "T_initial" in data_dict:
        data_dict["T_initial"] = data_dict["T_initial"].tolist()

    if "directional_weights" in data_dict and isinstance(
        data_dict["directional_weights"], np.ndarray
    ):
        data_dict["directional_weights"] = data_dict["directional_weights"].tolist()

    # Save in JSON
    with open(route_name, "w") as file:
        json.dump(data_dict, file)


def import_dict_json(route_name: str):
    """
    Function to import and transform Json file to dictionary,
    so that it can be re-used

    Args:
        route_name (str): full path name

    Returns:
        data (dict): information
    """

    with open(route_name, "r") as file:
        dictionary = dict(json.load(file))

    dictionary["input_matrix"] = jnp.array(dictionary["input_matrix"])

    # Cloud
    dictionary["cloud"] = MyCoordinates.mult_points(np.array(dictionary["cloud"]))

    # Output layout
    dictionary["output_layout"] = MyCoordinates.mult_points(
        np.array(dictionary["output_layout"])
    )

    return dictionary


def write_optimization_log(
    current_state: State,
    optimization_result,
    execution_time: float,
    cost_values: Array,
    show_results: bool,
    save_results: bool,
    results_file_name,
):
    """
    Function to generate a log containing all the relevant information from the
    optimisation. It can print through terminal or save all the information and
    set of plots.

    Args:
        current_state (class State): saving cloud, input_matrix(LxM), output_layout
                transcoding matrix shape and more
        optimization_result: containing all details of optimization process
        execution_time (float): time of execution of optimisation
        cost_values (jax array): values of cost in each iteration
        show_results: Flag to show results
        save_results: Flag to save results
        results_file_name: String of folder name where to save, if save_results=True

    """

    optimization_log: Dict[str, Any] = {}
    optimization_log["time_execution"] = execution_time
    optimization_log["n_it"] = optimization_result.nit
    optimization_log["success"] = optimization_result.success
    optimization_log["message"] = optimization_result.message
    optimization_log["final_cost_value"] = optimization_result.fun

    data = copy.deepcopy(current_state.__dict__)
    del data["cloud_points"]
    del data["output_layout"]
    del data["input_matrix"]
    del data["w"]
    data["initial_flatten_matrix"] = data["initial_flatten_matrix"].tolist()
    optimization_log["set_up"] = data
    optimization_log["optimised_transcoding_matrix"] = optimization_result.x.tolist()
    if "static_decoding_matrix" in optimization_log["set_up"].keys():
        static_decoding = optimization_log["set_up"]["static_decoding_matrix"]
        if type(static_decoding) is not int:
            optimization_log["set_up"][
                "static_decoding_matrix"
            ] = static_decoding.tolist()

    # Plot Optimization Process
    fig = plt.figure(figsize=(17, 9))

    # Plot 1 - Iterations vs Cost
    ax1 = fig.add_subplot(221)
    ax1.plot(range(len(cost_values)), cost_values)
    ax1.set_ylabel("Cost")
    ax1.set_xlabel("Number of iterations")
    ax1.set_title("Cost Vs. Iterations - BFGS")

    # Plot 2 - Cloud
    ax3 = fig.add_subplot(223)
    points = current_state.cloud_points.sph_deg()
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    m = Basemap(projection="robin", lon_0=0, resolution="c")
    x_map, y_map = m(x, y)
    m.drawcoastlines(linewidth=0.5, color="None")
    # draw parallels and meridians.
    m.drawparallels(
        np.arange(-90.0, 120.0, 45.0),
        labels=[True, True, False, False],
        labelstyle="+/-",
    )
    m.drawmeridians(
        np.arange(0.0, 360.0, 60.0),
        labels=[False, False, False, True],
        labelstyle="+/-",
    )
    ax3.scatter(
        x_map,
        y_map,
        s=50,
        c=z,
        cmap="coolwarm",
        alpha=0.8,
        edgecolors="none",
    )
    ax3.set_title("Cloud of points")

    # Plot 3 - Output Layout
    ax4 = fig.add_subplot(224)
    points = current_state.output_layout.sph_deg()
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    m = Basemap(projection="robin", lon_0=0, resolution="c")
    x_map, y_map = m(x, y)
    m.drawcoastlines(linewidth=0.5, color="None")
    # draw parallels and meridians.
    m.drawparallels(
        np.arange(-90.0, 120.0, 45.0),
        labels=[True, True, False, False],
        labelstyle="+/-",
    )
    m.drawmeridians(
        np.arange(0.0, 360.0, 60.0),
        labels=[False, False, False, True],
        labelstyle="+/-",
    )
    ax4.scatter(
        x_map,
        y_map,
        s=50,
        c=z,
        cmap="coolwarm",
        alpha=0.8,
        edgecolors="none",
    )
    ax4.set_title("Output layout")

    plt.subplots_adjust(hspace=1, wspace=0.5)

    if (save_results is True) and (type(results_file_name) is str):
        # Optimisation plot
        file_name = "optimization_plots.png"
        path = os.path.join("saved_results", results_file_name)
        full_path = os.path.join(path, file_name)
        os.makedirs(path, exist_ok=True)
        plt.savefig(full_path, dpi=300)

        # Optimisation log
        full_path = os.path.join(path, "optimization_log.json")
        with open(full_path, "w") as file:
            json.dump(optimization_log, file)

    if show_results:
        print("\n\n\n OPTIMIZATION LOG \n", json.dumps(optimization_log, indent=2))
        plt.show()


def save_results(info):
    """
    Function to that creates, if it is not already, the folder where to save data, and
    prepares the full path to call generate_json function

    Args:
        data (dict): information to be saved, it also contains where to save it
    """
    if type(info["results_file_name"]) is str:
        os.makedirs("saved_results", exist_ok=True)
        file_name = "optimization_info.json"
        path = os.path.join("saved_results", info["results_file_name"])
        full_path = os.path.join(path, file_name)
        os.makedirs(path, exist_ok=True)
        generate_json(info, full_path)
