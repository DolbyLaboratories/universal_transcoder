from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("white")
COLUMNS = ["azimuth", "elevation", "P", "V_r", "V_t", "E", "I_r", "I_t"]
PLOTS_PATH = Path(__file__).resolve().parents[1] / "saved_results"


def _get_path(name):
    return PLOTS_PATH / name / "signal_data.txt"


def box_plot(ut_txt_path, comp_txt_path, comp_name="comp", plot_type="pv"):

    ut_df = pd.read_csv(ut_txt_path, names=COLUMNS, header=None)
    ut_df["Method"] = "USAT"
    comp_df = pd.read_csv(comp_txt_path, names=COLUMNS, header=None)
    comp_df["Method"] = comp_name

    combined_dfs = pd.concat([ut_df, comp_df])

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
    sns.boxplot(
        data=df_long, x="Quantity", y="Value", hue="Method"
    )

    # Display the plot
    plt.show()


if __name__ == "__main__":
    box_plot(
        _get_path("5OAdecoding714_USAT"),
        _get_path("5OAdecoding714_allrad_maxre"),
        plot_type="ei",
        comp_name="AllRad"
    )
