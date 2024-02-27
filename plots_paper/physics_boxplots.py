from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Needed LaTeX packages: cm-super, type1cm, dvipng

# Set up font in the header of the file
# sns.set(style="ticks", font="Times New Roman", font)

plt.rc(
    "font",
    **{
        "family": "serif",
        "serif": ["Times New Roman"],
        "size": 9,
    }
)
plt.rc("axes", **{"labelsize": 9, "titlesize": 9})
plt.rc("text", usetex=True)
# If we use seaborn


COLUMNS = ["azimuth", "elevation", "P", "V_r", "V_t", "E", "I_r", "I_t"]
PLOTS_PATH = Path(__file__).resolve().parents[1] / "saved_results"
LATEX_NAMES = {
    "P": r"$P$",
    "V_r": r"$V^R$",
    "V_t": r"$V^T$",
    "E": r"$E$",
    "I_r": r"$I^R$",
    "I_t": r"$I^T$",
    "ASW": "ASW",
    "P_dB": r"$P$",
    "E_dB": r"$E$",
    "delta": r"$\delta$",
}


def _get_path(name):
    return PLOTS_PATH / name / "signal_data.txt"


def _save_plots(name):
    name_png = "boxplot_" + name.lower().replace(" ", "_") + ".png"
    name_pdf = "boxplot_" + name.lower().replace(" ", "_") + ".pdf"
    path_png = PLOTS_PATH / name_png
    path_pdf = PLOTS_PATH / name_pdf
    plt.savefig(path_pdf, bbox_inches="tight")
    plt.savefig(path_png, bbox_inches="tight")
    print("Plots saved in: %s" % PLOTS_PATH)


def filter_elevation(df, remove_negative_elevation=True):
    elevation = df["elevation"]
    mask = np.ones(len(elevation), dtype=bool)
    prob = 1 - np.cos(np.deg2rad(elevation.to_numpy()))  # Probability of removal
    prob_cumsum = np.cumsum(prob)
    prob_int = prob_cumsum.astype(int)
    mask[1:] = np.logical_not(prob_int[1:] - prob_int[:-1])
    if remove_negative_elevation:
        mask[elevation < 0] = False
    return df[mask]



def energy_to_db(energy):
    return 10 * np.log10(energy)


def pressure_to_db(pressure):
    return 10 * np.log10(pressure)


def radtrans_to_delta(iv_rad, iv_trans):
    return np.rad2deg(np.arctan2(iv_trans, iv_rad))


def radtrans_to_asw(iv_rad, iv_trans):
    iv = np.clip(np.sqrt(iv_rad**2 + iv_trans**2), 0, 1)
    return 3 / 8 * 2 * np.rad2deg(np.arccos(iv))


def box_plot(
    ut_txt_path,
    comp_txt_path,
    comp_name="comp",
    plot_type="pv",
    scale="human",
    title="",
    save_fig=True,
):
    ut_df = pd.read_csv(ut_txt_path, names=COLUMNS, header=None)
    ut_df["Method"] = "USAT"
    filter_elevation(ut_df)
    comp_df = pd.read_csv(comp_txt_path, names=COLUMNS, header=None)
    comp_df["Method"] = comp_name
    filter_elevation(comp_df)

    ref_df = pd.DataFrame(
        {
            "azimuth": [0],
            "elevation": [0],
            "P": [1],
            "V_r": [1],
            "V_t": [0],
            "E": [1],
            "I_r": [1],
            "I_t": [0],
            "Method": ["Ideal"],
        }
    )

    df = pd.concat([ref_df, ut_df, comp_df])

    # Selecting columns for the boxplot
    if scale == "linear":
        if plot_type == "pv":
            selected_columns = (["P", "V_r", "V_t"],)
        elif plot_type == "ei":
            selected_columns = (["E", "I_r", "I_t"],)
        else:
            raise ValueError("Wrong value for plot_type")
        unitlabels = ("",)
    elif scale == "human":
        if plot_type == "pv":
            df["P_dB"] = pressure_to_db(df["P"])
            df["ASW"] = radtrans_to_asw(df["V_r"], df["V_t"])
            df["delta"] = radtrans_to_delta(df["V_r"], df["V_t"])
            selected_columns = (["P_dB"], ["ASW", "delta"])
        elif plot_type == "ei":
            df["E_dB"] = energy_to_db(df["E"])
            df["ASW"] = radtrans_to_asw(df["I_r"], df["I_t"])
            df["delta"] = radtrans_to_delta(df["V_r"], df["V_t"])
            selected_columns = (["E_dB"], ["ASW", "delta"])
        else:
            raise ValueError("Wrong value for plot_type")

        unitlabels = ("dB", "deg")
    else:
        raise ValueError("Wrong value for plot_type")

    # Convert to longform
    df_long_list = [
        pd.melt(
            df,
            id_vars="Method",
            value_vars=sc,
            var_name="Quantity",
            value_name="Value",
        ).replace({"Quantity": LATEX_NAMES})
        for sc in selected_columns
    ]

    # Adding hue to the boxplot
    # Then when creating the figure
    mm = 1 / 25.4  # mm in inches
    width = 76 * mm  # Replace by 159 mm for 2-column plots
    aspect_ratio = 0.75  # Replace by whatever apprpriate
    fig, axs = plt.subplots(
        1,
        len(df_long_list),
        figsize=(width, aspect_ratio * width),
        width_ratios=[len(sc) for sc in selected_columns],
    )
    #plt.suptitle(title)

    for ax, df, unit in zip(axs, df_long_list, unitlabels):
        sns.boxplot(
            ax=ax,
            data=df,
            x="Quantity",
            y="Value",
            hue="Method",
            showfliers=False,
            dodge=True,
            fill=False,
        )

        ax.grid()
        ax.set_xlabel("")
        ax.set_ylabel(unit)
        ax.get_legend().remove()

    plt.subplots_adjust(bottom=0.4)
    lines, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        lines,
        labels,
        title=None,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.05),
        ncol=3,
        frameon=False,
        columnspacing=1,
        handletextpad=0.5,
    )
    plt.tight_layout()

    # Display the plot
    if save_fig:
        _save_plots(title)
    else:
        plt.show()


if __name__ == "__main__":
    box_plot(
        _get_path("5OAdecoding714_USAT"),
        _get_path("5OAdecoding714_allrad_maxre"),
        plot_type="ei",
        comp_name="AllRad",
        title="Decoding 5OA to 7.0.4",
    )
    box_plot(
        _get_path("704transcoding5OA_USAT"),
        _get_path("704transcoding5OA_direct"),
        plot_type="pv",
        comp_name="HOA enc.",
        title="Transcoding 7.0.4 to 5OA",
    )
    box_plot(
        _get_path("ex_decoding_301_irregular"),
        _get_path("ex_decoding_301irregular_vbap.png"),
        plot_type="ei",
        comp_name="Remapping",
        title="Decoding 5.0.2 to 3.0.1 irregular",
    )
    box_plot(
        _get_path("panning51_USAT"),
        _get_path("panning51_direct"),
        plot_type="ei",
        comp_name="Tangent law",
        title="Rendering to 5.1",
    )
