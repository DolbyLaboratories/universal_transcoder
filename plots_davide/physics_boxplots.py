from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set up font in the header of the file
from matplotlib import rc
rc(
    "font",
    **{
        "family": "serif",
        "serif": ["Times"],
        "size": 9,  # Equivalent to \small in LaTeX
    }
)
rc("text", usetex=True)
sns.set(style="ticks", font="Times New Roman")   # If we use seaborn


COLUMNS = ["azimuth", "elevation", "P", "V_r", "V_t", "E", "I_r", "I_t"]
PLOTS_PATH = Path(__file__).resolve().parents[1] / "saved_results"


def _get_path(name):
    return PLOTS_PATH / name / "signal_data.txt"


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


def box_plot(ut_txt_path, comp_txt_path, comp_name="comp", plot_type="pv", title=""):

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

    combined_dfs = pd.concat([ref_df, ut_df, comp_df])

    # Selecting columns for the boxplot
    if plot_type == "pv":
        selected_columns = ["P", "V_r", "V_t"]
    elif plot_type == "ei":
        selected_columns = ["E", "I_r", "I_t"]
    else:
        raise ValueError("Wrong value for plot_type")

    # Convert to longform
    df_long = pd.melt(
        combined_dfs,
        id_vars="Method",
        value_vars=selected_columns,
        var_name="Quantity",
        value_name="Value",
    )

    # Adding hue to the boxplot

    # Then when creating the figure
    mm = 1 / 25.4  # mm in inches
    width = 76 * mm   # Replace by 159 mm for 2-column plots
    aspect_ratio = 1.3  # Replace by whatever apprpriate
    fig, ax = plt.subplots(figsize=(width, aspect_ratio*width))
    ax = sns.boxplot(
        ax=ax,
        data=df_long,
        x="Quantity",
        y="Value",
        hue="Method",
        showfliers=False,
        dodge=True,
        fill=False,
    )
    ax.set_title(title)

    plt.legend(title=None, loc="upper right")  # bbox_to_anchor=(1.05, 1),
    plt.grid()

    if plot_type == "pv":
        ax.set_xticklabels([r"$P$", r"$V_r$", r"$V_t$"])
    elif plot_type == "ei":
        ax.set_xticklabels([r"$E$", r"$I_r$", r"$I_t$"])

    plt.subplots_adjust(bottom=0.3)
    plt.legend(title=None, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=1, frameon=False)
    plt.tight_layout()


    # Display the plot


    plt.show()


if __name__ == "__main__":
    box_plot(
        _get_path("5OAdecoding714_USAT"),
        _get_path("5OAdecoding714_allrad_maxre"),
        plot_type="ei",
        comp_name="AllRad",
        title="5th order HOA decoding to 7.0.4",
    )
    box_plot(
        _get_path("704transcoding5OA_USAT"),
        _get_path("704transcoding5OA_direct"),
        plot_type="pv",
        comp_name="Remapping (VBAP)",
        title="Transcoding 7.0.4 to 5th order HOA",
    )
    box_plot(
        _get_path("ex_decoding_301_irregular"),
        _get_path("ex_decoding_301irregular_vbap.png"),
        plot_type="ei",
        comp_name="Remapping (VBAP)",
        title="Decoding 5.0.2 to a 3.0.1 irregular layout",
    )
    box_plot(
        _get_path("panning51_USAT"),
        _get_path("panning51_direct"),
        plot_type="ei",
        comp_name="Tangent law / VBAP",
        title="Rendering to 5.1",
    )
