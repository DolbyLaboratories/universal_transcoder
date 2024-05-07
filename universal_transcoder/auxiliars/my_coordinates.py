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

import numpy as np
import pyfar as pf

from universal_transcoder.auxiliars.typing import NpArray


class MyCoordinates(pf.Coordinates):
    """Class generated out of pyfar.Coordinates to extents its capabilities.
    Pyfar.Coordinates allows inputs belonging to [0º,360º] for azimut angle
    and [-90º,90º] for elevation. The main goal of the following methods is
    to allow azimut inputs of [-180º, 180º]. Additionally, there are methods to
    retreive the angles information in the above mentioned convention"""

    # Create Single Points #
    @classmethod
    def point(cls, theta, phi, d=1):
        if (theta > 180) or (theta < (-180)) or (phi > 90) or (phi < (-90)):
            raise ValueError(
                "Azimut position must be [-180,180] and elevation position must be [-90,90]"
            )
        if theta < 0:  # theta [-180,0] --> [0,360]
            theta = theta + 360
        point = cls(theta, phi, d, domain="sph", convention="top_elev", unit="deg")
        return point

    # Create Multi-points #
    @classmethod
    def mult_points(cls, positions: NpArray):
        thetas = positions[:, 0]
        phis = positions[:, 1]
        ds = positions[:, 2]
        # check correct format
        if (
            (np.any(thetas > 180))
            or (np.any(thetas < (-180)))
            or (np.any(phis > 90))
            or (np.any(phis < (-90)))
        ):
            raise ValueError(
                "Azimut position must be [-180,180] and elevation position must be [-90,90]"
            )
        # check correct order (ascending thetas)
        # sorted = np.all(np.diff(thetas) >= 0)
        # if sorted == False:
        #     raise ValueError(
        #         "Positions must be introduced in ascending order of azimut angle"
        #     )

        # put thetas [0,360] for pf.Coordinates
        for i in range(thetas.shape[0]):
            if thetas[i] < 0:
                thetas[i] = thetas[i] + 360

        # generate Coordinates space
        mult_points = cls(
            thetas, phis, ds, domain="sph", convention="top_elev", unit="deg"
        )
        return mult_points

    # Create Multi-points - CARTESIAN COORDINATES #
    @classmethod
    def mult_points_cart(cls, positions: NpArray):
        x = positions[:, 0]
        y = positions[:, 1]
        z = positions[:, 2]

        # generate Coordinates space
        mult_points = cls(x, y, z, domain="cart", convention="right")
        return mult_points

    # Get (Multi)Point coordinates: Polar in Degrees or Radians; and Cartesian
    def sph_deg(self):
        sph_coord = self.get_sph(convention="top_elev", unit="deg")
        for i in range(sph_coord.shape[0]):
            if sph_coord[i][0] > 180:
                sph_coord[i][0] = sph_coord[i][0] - 360
        return sph_coord

    def sph_rad(self):
        sph_coord = self.get_sph(convention="top_elev", unit="rad")
        for i in range(sph_coord.shape[0]):
            if sph_coord[i][0] > np.pi:
                sph_coord[i][0] = sph_coord[i][0] - 2 * np.pi
        return sph_coord

    def cart(self):
        cart_coord = self.get_cart(convention="right")
        return cart_coord

    def discard_lower_hemisphere(self):
        """
        Remove the lower hemisphere points from the cloud
        Returns:
            New cloudd of points with lower hemisphere removed
        """
        return self[self.cart()[:, 2] >= 0]
