import os

import matplotlib.pyplot as pyp
import numpy as np

eps = np.finfo(float).eps


def get_test_points():
    el_min_max = (-90, 90)
    az_min_max = (-180, 180)
    elevation = range(min(el_min_max), max(el_min_max))
    azimuth = range(min(az_min_max), max(az_min_max))
    elevation = [angle + 0.5 for angle in elevation]
    azimuth = [angle + 0.5 for angle in azimuth]

    phi_test_rad = np.array(
        [[angles / 180.0 * np.pi for angles in azimuth] for el in elevation]
    )
    theta_test_rad = np.array(
        [[angles / 180.0 * np.pi for angles in elevation] for az in azimuth]
    )

    az_grad = np.array([[angles for angles in azimuth] for el in elevation])
    el_grad = np.array([[angles for angles in elevation] for az in azimuth])
    return (
        az_grad.reshape(-1),
        el_grad.T.reshape(-1),
        phi_test_rad.reshape(-1),
        theta_test_rad.T.reshape(-1),
    )


def interpolate(start, end, npoints):
    interpolated = np.array(
        [x / npoints * (end - start) + start for x in range(0, npoints)]
    )
    return interpolated


def radial(A, B):
    Sum = 0
    if len(A) != len(B):
        raise ValueError("Arrays with different shapes. [Radial functionsh.py]")

    for i in range(len(A)):
        Sum += A[i] * B[i]
    return Sum


def tang(A, B):  # aka Cross
    Sum = 0
    if len(A) != len(B):
        raise ValueError("Arrays with different shapes. [Tang functionsh.py]")
    for i in range(len(A)):
        Sum += (A[i] * B[i - 1] - A[i - 1] * B[i]) ** 2.0
    return np.sqrt(Sum)


def velocity_intensity(phi_test, data, spk_azimuth, spk_elevation, theta_test=None):
    spk_x = np.cos(spk_azimuth / 180.0 * np.pi) * np.cos(spk_elevation / 180.0 * np.pi)
    spk_y = np.sin(spk_azimuth / 180.0 * np.pi) * np.cos(spk_elevation / 180.0 * np.pi)
    spk_z = np.sin(spk_elevation / 180.0 * np.pi)

    y = data.T
    Vx = np.sum((y.T * spk_x).T, axis=0) / (np.sum(y, axis=0) + eps)
    Vy = np.sum((y.T * spk_y).T, axis=0) / (np.sum(y, axis=0) + eps)
    Vz = np.sum((y.T * spk_z).T, axis=0) / (np.sum(y, axis=0) + eps)
    V = Vx, Vy, Vz

    sq = y * y
    Jx = np.sum((sq.T * spk_x).T, axis=0) / (np.sum(sq, axis=0) + eps)
    Jy = np.sum((sq.T * spk_y).T, axis=0) / (np.sum(sq, axis=0) + eps)
    Jz = np.sum((sq.T * spk_z).T, axis=0) / (np.sum(sq, axis=0) + eps)
    J = Jx, Jy, Jz

    phi = np.array(phi_test)
    if theta_test is None:
        theta_test = np.zeros(phi.shape)
    test_x = np.cos(phi / 180.0 * np.pi) * np.cos(theta_test / 180.0 * np.pi)
    test_y = np.sin(phi / 180.0 * np.pi) * np.cos(theta_test / 180.0 * np.pi)
    test_z = np.sin(theta_test / 180.0 * np.pi)
    test = test_x, test_y, test_z

    V_radial = radial(V, test)
    J_radial = radial(J, test)
    V_tang = tang(V, test)
    J_tang = tang(J, test)
    return V_radial, V_tang, J_radial, J_tang


def save_matrix_to_file(gains_matrix, filename):
    np.savetxt(filename, gains_matrix, fmt="%.6f", delimiter=", ", newline="\n")
    return


def ambi_decode(allrad_deco, ambi_enco, filename):
    spk_gains = allrad_deco @ ambi_enco
    save_matrix_to_file(spk_gains, filename)
    return spk_gains


def save_physics_to_file(
    position_az, position_el, p, V_radial, V_tang, E, J_radial, J_tang, filename
):
    # Correct directory
    os.makedirs("saved_results", exist_ok=True)
    txt_name = "signal_data.txt"
    path = os.path.join("saved_results", filename)
    full_path = os.path.join(path, txt_name)
    os.makedirs(path, exist_ok=True)
    fh = open(full_path, "w")
    # Organise data
    for idx, az in enumerate(position_az):
        str_phy = "%.4f, %.4f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f\n" % (
            position_az[idx],
            position_el[idx],
            p[idx],
            V_radial[idx],
            V_tang[idx],
            E[idx],
            J_radial[idx],
            J_tang[idx],
        )
        fh.write(str_phy)

    fh.close()
    return


def get_physics(gains, phi_test_grad, theta_test_grad, spk_azimuth, spk_elevation):
    p = np.sum(gains, 1)
    E = np.sum(gains**2, 1)

    phi_test_grad = np.array(phi_test_grad).reshape(-1)
    theta_test_grad = np.array(theta_test_grad).reshape(-1)
    V_radial, V_tang, J_radial, J_tang = velocity_intensity(
        phi_test_grad,
        np.array(gains),
        np.array(spk_azimuth),
        np.array(spk_elevation),
        theta_test=theta_test_grad,
    )
    return p, V_radial, V_tang, E, J_radial, J_tang


def gains_plot(
    phi_test,
    data,
    chlabels,
    title,
    extension=".png",
    dB=False,
    plot_folder="",
    spk_azimuth=None,
    spk_elevation=None,
    pressure=False,
    theta_test=None,
):
    pyp.figure(figsize=(10, 6))
    ax = pyp.subplot(111)
    labels = chlabels.copy()

    if not labels:
        labels = [str(idx + 1) for idx in range(data.shape[1])]

    color_idx = np.linspace(0, 1, data.shape[1])
    colors = pyp.cm.viridis(color_idx)

    pressure_sum = np.sum(data, 1)
    energy_sum = np.sum(data**2, 1)

    if spk_azimuth is None:
        spk_azimuth = np.array(spk_azimuth)
        spk_elevation = np.array(spk_elevation)

    V_radial, V_tang, J_radial, J_tang = velocity_intensity(
        phi_test, data, spk_azimuth, spk_elevation, theta_test=theta_test
    )

    if dB:
        J_radial = 20 * np.log10(abs(J_radial))
        J_tang = 20 * np.log10(abs(J_tang))

        data = 10 * np.log10(data**2)
        pressure_sum = 20 * np.log10(abs(pressure_sum))
        energy_sum = 10 * np.log10(energy_sum)

    ax.set_xlabel("Angle (deg)")
    ax.set_ylabel("Gains (linear)")
    ax.set_title(title)
    ax.grid(linestyle=":")
    ticks = [-180, -120, -60, 0, 60, 120, 180]
    pyp.xticks(ticks)

    if dB:
        pyp.ylim(-50, 6)
        title += "_dB"
        ax.set_ylabel("Gains (dB)")
    else:
        pyp.ylim(-0.2, 1.2)
    pyp.xlim(-180, 180)

    angles = [360.0 / len(phi_test) * idx for idx in range(len(phi_test))]
    angles = np.array(angles) - 180
    if pressure:
        pyp.plot(angles, pressure_sum, color="b", linewidth=4)

    # decoding to speakers
    if dB:
        pyp.plot(angles, energy_sum, color="k", linewidth=4)
    if not dB and not pressure:
        pyp.plot(angles, J_radial, color="r", linewidth=3, linestyle="-.")
    if not dB and not pressure:
        pyp.plot(angles, J_tang, color="y", linewidth=3, linestyle="-.")
    if not dB and pressure:
        pyp.plot(angles, V_radial, color="r", linewidth=3, linestyle="-.")
    if not dB and pressure:
        pyp.plot(angles, V_tang, color="y", linewidth=3, linestyle="-.")

    horiz_spk_mask = spk_elevation == 0
    num_horiz_speakers = np.sum(horiz_spk_mask)
    num_non_horiz_speakers = spk_elevation.size - num_horiz_speakers

    horiz_counter = 0
    non_horiz_counter = 0
    for idx in range(data.shape[1]):
        # if idx < 7: # plot horiz gains with continuous line
        if horiz_spk_mask[idx]:
            colors = pyp.cm.viridis(np.linspace(0, 1, num_horiz_speakers))
            pyp.plot(
                angles,
                data[:, idx],
                linewidth=2,
                color=colors[horiz_counter].tolist(),
            )
            horiz_counter += 1
        else:
            colors = pyp.cm.viridis(np.linspace(0, 1, num_non_horiz_speakers))
            pyp.plot(
                angles,
                data[:, idx],
                linestyle="--",
                linewidth=2,
                color=colors[non_horiz_counter].tolist(),
            )
            non_horiz_counter += 1

    if not dB and not pressure:
        labels.insert(0, "transverse intensity")
        labels.insert(0, "radial intensity")
    if dB:
        labels.insert(0, "energy")

    if not dB and pressure:
        labels.insert(0, "transverse velocity")
        labels.insert(0, "radial velocity")
    if pressure:
        labels.insert(0, "pressure")

    # legend outside the plot
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    legend_x = 1
    legend_y = 0.5
    pyp.legend(labels, loc="center left", bbox_to_anchor=(legend_x, legend_y))

    pyp.savefig(os.path.join(plot_folder, title + extension))
    ax.set_title("")
    pyp.savefig(
        os.path.join(plot_folder, title + ".pdf"), bbox_inches="tight", pad_inches=0
    )
    pyp.close()


def make_txt_and_plots(
    spk_gains, spk_azimuth, spk_elevation, plot_folder=None, ch_labels=None
):
    if plot_folder is None:
        plot_folder = "data_for_paper"
    if ch_labels is None:
        ch_labels = [str(idx) for idx in range(len(spk_elevation))]

    os.makedirs(plot_folder, exist_ok=True)

    # # save matrix to file ...
    # remeber az, el, p, v, e, i
    # in this order
    az_grad, el_grad, phi_test_rad, theta_test_rad = get_test_points()

    pressure, V_radial, V_tang, Energy, J_radial, J_tang = get_physics(
        spk_gains.T, az_grad, el_grad, spk_azimuth, spk_elevation
    )
    out_filename = os.path.join(plot_folder, "ambi_704_physics.txt")
    save_physics_to_file(
        az_grad,
        el_grad,
        pressure,
        V_radial,
        V_tang,
        Energy,
        J_radial,
        J_tang,
        out_filename,
    )

    title = "decoded_ambi_gains_74_"
    hv_str = "horizontal_allrad_energy"
    gains_plot(
        az_grad[el_grad == 0.5],
        spk_gains[:, el_grad == 0.5].T,
        ch_labels,
        title + hv_str,
        theta_test=None,
        plot_folder=plot_folder,
        spk_azimuth=spk_azimuth,
        spk_elevation=spk_elevation,
    )
    gains_plot(
        az_grad[el_grad == 0.5],
        spk_gains[:, el_grad == 0.5].T,
        ch_labels,
        title + hv_str,
        theta_test=None,
        plot_folder=plot_folder,
        dB=True,
        spk_azimuth=spk_azimuth,
        spk_elevation=spk_elevation,
    )

    hv_str = "horizontal_allrad_pressure"
    gains_plot(
        az_grad[el_grad == 0.5],
        spk_gains[:, el_grad == 0.5].T,
        ch_labels,
        title + hv_str,
        theta_test=None,
        plot_folder=plot_folder,
        spk_azimuth=spk_azimuth,
        spk_elevation=spk_elevation,
        pressure=True,
    )
    gains_plot(
        az_grad[el_grad == 0.5],
        spk_gains[:, el_grad == 0.5].T,
        ch_labels,
        title + hv_str,
        theta_test=None,
        plot_folder=plot_folder,
        dB=True,
        spk_azimuth=spk_azimuth,
        spk_elevation=spk_elevation,
        pressure=True,
    )
