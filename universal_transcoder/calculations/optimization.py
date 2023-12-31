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

import os
import time
import jax
import jax.numpy as jnp
import numpy as np
from jax import grad
from scipy.optimize import minimize
from universal_transcoder.calculations.cost_function import State
from universal_transcoder.calculations.set_up_system import (
    set_up_general,
)
from universal_transcoder.plots_and_logs.all_plots import (
    plots_general,
)
from universal_transcoder.plots_and_logs.write_logs import (
    write_optimization_log,
    save_results,
)

os.environ["JAX_ENABLE_X64"] = "1"


def optimize(info: dict):
    """
    Complete optimization process. Function to prepare the optimization, to call the
    optimization obtaining the final optimised decoding matrix (NxM) for each given situation.
    Depending on the activated flags it also generates, saves or shows interesting information
    regarding the optimization process and the results.

    Args:
        info (dict): All the info needed for the optimization given as the following:
            dictionary = {
                "input_matrix_optimization": input,     # Input matrix that encodes in input format LxM
                "cloud_optimization": cloud,            # Cloud of points sampling the sphere (L)
                "output_layout": output_layout,         # Output layout of speakers to decode
                "coefficients": {                       # List of coefficients to the cost function
                    "energy": 0,
                    "radial_intensity": 0,
                    "transverse_intensity": 0,
                    "pressure": 0,
                    "radial_velocity": 0,
                    "transverse_velocity": 0,
                    "in_phase_lin": 0,
                    "symmetry_lin": 0,
                    "total_gains_quad": 0,
                    "in_phase_quad": 0,
                    "symmetry_quad": 0,
                    "total_gains_quad": 0,
                },
                "directional_weights": 1,               # Weights given to different directions sampling the sphere
                "show_results": True,                   # Flag to show results
                "save_results": False,                  # Flag to save results
                "cloud_plots": cloud,                   # Cloud of points sampling the sphere for the plots(P)
                "input_matrix_plots": matrix,           # Matrix that encodes in input format PxM, for the plots
                "results_file_name": "name",            # String of folder name where to save, if save_results=True
            }


    Returns:
        D_optimized (numpy Array complex64): optimized decoding matrix (NxM) that decodes
        from the input format to the output format
    """
    # Save optimisation data in json
    if info["save_results"]:
        save_results(info)

    ## Set up optimisation
    current_state, D_flatten_initial = set_up_general(info)

    # Call optimization function
    D_flatten_optimized = bfgs_optim(
        current_state,
        D_flatten_initial,
        info["show_results"],
        info["save_results"],
        info["results_file_name"],
    )

    # Adapt / reshape result of optimization --> Decoding matrix
    D_optimized = np.array(D_flatten_optimized).reshape(
        current_state.decoding_matrix_shape
    )

    # If show or save flags active
    if info["show_results"] or info["save_results"]:
        if "cloud_plots" in info:
            plots_general(
                info["output_layout"],
                D_optimized,
                info["input_matrix_plots"],
                info["cloud_plots"],
                info["show_results"],
                info["save_results"],
                info["results_file_name"],
            )
        else:
            plots_general(
                info["output_layout"],
                D_optimized,
                info["input_matrix_optimization"],
                info["cloud_optimization"],
                info["show_results"],
                info["save_results"],
                info["results_file_name"],
            )

    return D_optimized


def bfgs_optim(
    current_state: State,
    flatten_initial_dec: jnp,
    show_results: bool,
    save_results: bool,
    results_file_name,
):
    """
    Optimization function to generate an optimized flatten decoder matrix using
    BFGS method. It can also print through terminal or save an optimisation log

    Args:
        current_state (class State): saving cloud, input_matrix(LxM), output_layout
                and decoder_matrix shape
        flatten_initial_dec (numpy Array): not-optimized flatten decoding matrix from
                input format to output_layout ((NxM)x1 size)
        show_results (bool): flag to show plots and results
        save_results (bool): flag to save plots and results
        results_file_name (String or None): in case save_results=True, then it is the
                String that gives name to the folder where results are saved

    Returns:
        dec_matrix_bfgs (numpy Array): optimized flatten decoding matrix ((NxM)x1 size)
    """

    # Initial time
    start_time = time.time()

    # Optimization
    cost_values = []
    cost_values.append(
        current_state.cost_function(flatten_initial_dec)
    )  # starting point

    # Function to save cost value in each iteration
    def callback_func(xk):
        cost_values.append(current_state.cost_function(xk))

    # Include jit, reduces execution time
    cost_function_jit = jax.jit(current_state.cost_function)

    # Actual optimisation
    optimization_result = minimize(
        cost_function_jit,  # current_state.cost_function, if jit unwanted
        flatten_initial_dec,
        jac=grad(cost_function_jit),  # current_state.cost_function , if jit unwanted
        method="BFGS",
        callback=callback_func,
    )

    # Final flatten optimised matrix
    dec_matrix_bfgs = optimization_result.x

    # End time
    execution_time = time.time() - start_time

    # Optimization Log
    if show_results or save_results:
        write_optimization_log(
            current_state,
            optimization_result,
            execution_time,
            cost_values,
            show_results,
            save_results,
            results_file_name,
        )

    return dec_matrix_bfgs
