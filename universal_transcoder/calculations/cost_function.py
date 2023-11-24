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

import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
from universal_transcoder.auxiliars.get_left_right_pairs import get_left_right_pairs
from universal_transcoder.auxiliars.my_coordinates import MyCoordinates
from universal_transcoder.calculations.energy_intensity import (
    transverse_I_calculation,
    energy_calculation,
    radial_I_calculation,
)
from universal_transcoder.calculations.pressure_velocity import (
    pressure_calculation,
    transversal_V_calculation,
    radial_V_calculation,
)


@dataclass
class State:
    """Class to store all neccessary side-data for optimization. It includes the
    cost function itself and some other methods needed for the optimization process.
    Main functions are documented below."""

    cloud_points: MyCoordinates
    input_matrix: jnp
    output_layout: MyCoordinates
    decoding_matrix_shape: tuple = (0, 0)
    initial_flatten_matrix: jnp = 0
    ce: float = 0  # Energy Coefficient (quadratic)
    cir: float = 0  # Radial Intensity Coefficient (quadratic)
    cit: float = 0  # Transverse Intensity Coefficient (quadratic)
    cp: float = 0  # Pressure Coefficient (linear)
    cvr: float = 0  # Radial Velocity Coefficient (linear)
    cvt: float = 0  # Transverse Velocity Coefficient (linear)
    cphquad: float = 0  # In-Phase Coefficient (quadratic)
    csymquad: float = 0  # Symmetry Coefficient (quadratic)
    cphlin: float = 0  # In-Phase Coefficient (linear)
    csymlin: float = 0  # Symmetry Coefficient (linear)
    cminglin: float = 0  # Total gains Coefficient (linear)
    cmingquad: float = 0  # Total gains Coefficient (quadratic)
    w: np = 1
    n: int = 1

    def cost_function(self, flatten_decoding_matrix: jnp):
        """Cost function. Calculates the cost value

        Args:
           flatten_decoding_matrix: input vector, in which first half corresponds
           to the real part and second half to the imaginary part of the matrix

        Returns:
            cost value (float): value of the resulting cost
        """

        # Reshape and re-construct decoding_matrix
        decoding_matrix = jnp.reshape(
            flatten_decoding_matrix, self.decoding_matrix_shape
        )

        # Energy vector
        energy = energy_calculation(self.input_matrix, decoding_matrix)
        # Radial intensity vector
        radial_i = radial_I_calculation(
            self.cloud_points, self.input_matrix, self.output_layout, decoding_matrix
        )

        # Transversal intensity vector
        transverse_i = transverse_I_calculation(
            self.cloud_points, self.input_matrix, self.output_layout, decoding_matrix
        )
        # Pressure vector
        pressure = pressure_calculation(self.input_matrix, decoding_matrix)
        # Radial velocity vector
        radial_v = radial_V_calculation(
            self.cloud_points, self.input_matrix, self.output_layout, decoding_matrix
        )
        # Transversal velocity vector
        transverse_v = transversal_V_calculation(
            self.cloud_points, self.input_matrix, self.output_layout, decoding_matrix
        )
        # Output gains and in_phase vector
        output_gains = jnp.dot(self.input_matrix, decoding_matrix.T)
        in_phase_quad, in_phase_lin = self._in_phase_1(output_gains)
        # Symmetry vector
        symmetric_gains = self._rearrange_gains(
            self.output_layout, self.cloud_points, output_gains
        )
        asymmetry_quad, asymmetry_lin = self._symmetry_differences(
            output_gains, symmetric_gains
        )
        asymmetry_quad = jnp.real(asymmetry_quad)
        asymmetry_lin = jnp.real(asymmetry_lin)
        # Minimize gains
        total_gains_lin = jnp.sum(
            jnp.abs(flatten_decoding_matrix)
            > (10 ** (3.0 / 20))  # try to limit boost to X dBs
        )  # try to stay below +6dB of gain
        total_gains_quad = jnp.sum(
            jnp.abs(flatten_decoding_matrix) ** 2
            > (10 ** (3.0 / 10))  # try to limit boost to X dBs
        )  # try to stay below +6dB of gain

        # Calculate individual cost functions
        self.n = jnp.size(energy)
        C_e = self._quadratic_avg(1.0 - energy)
        C_ir = self._quadratic_avg(1.0 - radial_i)
        C_it = self._quadratic_avg(transverse_i)
        C_p = self._quadratic_avg(1.0 - pressure)
        C_vr = self._quadratic_avg(1.0 - radial_v)
        C_vt = self._quadratic_avg(transverse_v)
        C_ph_quad = self._quadratic_avg(in_phase_quad)
        C_sym_quad = self._quadratic_avg(asymmetry_quad)
        C_ph_lin = self._quadratic_avg(in_phase_lin)
        C_sym_lin = self._quadratic_avg(asymmetry_lin)
        C_min_gains_lin = total_gains_lin
        C_min_gains_quad = total_gains_quad

        self._print(
            C_e,
            C_ir,
            C_it,
            C_p,
            C_vr,
            C_vt,
            C_ph_quad,
            C_sym_quad,
            C_ph_lin,
            C_sym_lin,
            C_min_gains_lin,
            C_min_gains_quad,
        )

        # Calculate complete cost value
        cost_value = (
            self.ce * C_e
            + self.cir * C_ir
            + self.cit * C_it
            + self.cp * C_p
            + self.cvr * C_vr
            + self.cvt * C_vt
            + self.cphquad * C_ph_quad
            + self.csymquad * C_sym_quad
            + self.cphlin * C_ph_lin
            + self.csymlin * C_sym_lin
            + self.cminglin * C_min_gains_lin
            + self.cmingquad * C_min_gains_quad
        )

        return cost_value

    def _quadratic_avg(self, variable):
        return (1.0 / self.n) * jnp.sum((variable**2) * self.w)

    @staticmethod
    def _in_phase_1(output_gains):
        """Function that calculates the not-in-phase contribution, accounting those gains which
        are negative

        Args:
           output_gains (numpy Array): array of output gains of each of the N speakers to
                recreate each of the L virtual directions sampling the sphere. Shape LxN

        Returns:
            in_phase_quad (numpy Array): quadratic not-in-phase contribution of the set of
                    N speakers when reproducing each of the L virtual sources . Shape 1xL
            in_phase_lin (numpy Array): linear not-in-phase contribution of the set of
                    N speakers when reproducing each of the L virtual sources . Shape 1xL
        """
        output_gains_R = jnp.real(output_gains)
        mask = jnp.heaviside(-output_gains_R, 0)
        in_phase_quad = jnp.sum((output_gains_R * mask) ** 2, axis=1)
        in_phase_lin = jnp.sum((abs(output_gains_R) * mask), axis=1)
        return in_phase_quad, in_phase_lin

    @staticmethod
    def _rearrange_gains(layout: MyCoordinates, cloud: MyCoordinates, gains: jnp):
        """Generates a rearranged copy of an input array of gains so that the gains
        of symmetric pairs are coincident in the same position

        Args:
           layout (MyCoordinates): set of positions of speakers in layout
           cloud(MyCoordinates): set of positions of directions sampling the sphere
           gains (numpy Array): array of gains to be rearranged

        Returns:
            final_gains (numpy Array): same shape as input gains, but rearranged
        """
        pairs_layout = get_left_right_pairs(layout)
        pairs_cloud = get_left_right_pairs(cloud)

        N = pairs_layout.size
        L = pairs_cloud.size

        # first I get the order for columns
        gains_columns = np.zeros(N).astype(int)
        for i in range(N):
            aux = pairs_layout[i]
            if aux == 0:
                gains_columns[i] = i
            else:
                my_pair = np.where(pairs_layout == -aux)[0][0]
                gains_columns[i] = int(my_pair)

        # now I get the order for rows
        gains_rows = np.zeros(L).astype(int)
        for i in range(L):
            aux = pairs_cloud[i]
            if aux == 0:
                gains_rows[i] = i
            else:
                my_pair = np.where(pairs_cloud == -aux)[0][0]
                gains_rows[i] = int(my_pair)

        # apply re-ordering
        aux_gains = jnp.zeros(gains.shape)
        final_gains = jnp.zeros(gains.shape)

        aux_gains = aux_gains.at[:, gains_columns].set(gains)
        final_gains = final_gains.at[gains_rows, :].set(aux_gains)

        return final_gains

    @staticmethod
    def _symmetry_differences(gains: jnp, reordered_gains: jnp):
        """Function that calculates the asymmetry contribution, receiving as input two arrays,
        in which coincident positions correspond to the gains of symmetric pairs.

        Args:
            gains (numpy Array): array of output gains of each of the N speakers to
                recreate each of the L virtual directions sampling the sphere. Shape LxN
           reordered_gains (numpy Array): re-arranged array of output gains of each of the
                N speakers to recreate each of the L virtual directions sampling the sphere,
                in such order that each position corresponds to the symmetric pair of the
                other input array "gains". Shape LxN

        Returns:
            asymmetry_quad (numpy Array): quadratic asymmetry contribution of the set of
                    N speakers when reproducing each of the L virtual sources . Shape 1xL
            asymmetry_lin (numpy Array): linear asymmetry contribution of the set of
                    N speakers when reproducing each of the L virtual sources . Shape 1xL
        """
        differences = gains - reordered_gains
        asymmetry_quad = jnp.sum(jnp.abs(differences) ** 2, axis=1)
        asymmetry_lin = jnp.sum(jnp.abs(differences), axis=1)
        return asymmetry_quad, asymmetry_lin

    def reset_state(self):
        self.ce = 0
        self.cir = 0
        self.cit = 0
        self.cp = 0
        self.cvr = 0
        self.cvt = 0
        self.cphlin = 0
        self.csymlin = 0
        self.cphquad = 0
        self.csymquad = 0
        self.cminglin = 0
        self.cmingquad = 0
        return

    def _print(
        self,
        C_e,
        C_ir,
        C_it,
        C_p,
        C_vr,
        C_vt,
        C_ph_quad,
        C_sym_quad,
        C_ph_lin,
        C_sym_lin,
        C_min_gains_lin,
        C_min_gains_quad,
    ):
        print(
            "Costs: \n Pressure ",
            C_p,
            "  Veloc.Rad ",
            C_vr,
            " Veloc.Trans ",
            C_vt,
            " energy ",
            C_e,
            " Inten.Rad ",
            C_ir,
            " Inten.Trans ",
            C_it,
            " In phase (quad)",
            C_ph_quad,
            " Symmetry (quad)",
            C_sym_quad,
            " In phase (lin)",
            C_ph_lin,
            " Symmetry (lin)",
            C_sym_lin,
            " Total gains (lin)",
            C_min_gains_lin,
            " Total gains (quad)",
            C_min_gains_quad,
        )
