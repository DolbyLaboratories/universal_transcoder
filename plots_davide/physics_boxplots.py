import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


ut_txt_filename = "saved_results/paper_case2_ambi5OAto704_allrad_maxre/signal_data.txt"
comp_txt_filename = (
    "plots_davide/allrad_basic_renorm_zotter_n3d_mean/ambi_704_physics.txt"
)

columns = ["azimuth", "elevation", "p", "v_r", "v_t", "e", "i_r", "i_t"]
ut_df = pd.read_csv(ut_txt_filename, names=columns, header=None)
comp_df = pd.read_csv(comp_txt_filename, names=columns, header=None)

combined_dfs = pd.DataFrame(
    {
        "ut": ut_df["p"],
        "comp": comp_df["p"],
    }
)

plt.figure()
sns.set_style("white")
sns.boxplot(data=combined_dfs, palette="viridis")
plt.xlabel("encoding/decoding method")
plt.ylabel("pressure")
plt.show()

# example of...
# putting things together
cut_mask = ut_df["elevation"] > -0.5
ut_df_cut = ut_df[cut_mask]
cut_mask = comp_df["elevation"] > -0.5
comp_df_cut = comp_df[cut_mask]
combined_dfs = pd.DataFrame(
    {
        "ut pressure": ut_df_cut["p"],
        "comp pressure": comp_df_cut["p"],
        "ut energy": ut_df_cut["e"],
        "comp energy": comp_df_cut["e"],
        "ut intensity": ut_df_cut["i_r"],
        "comp intensity": comp_df_cut["i_r"],
    }
)

plt.figure()
sns.set_style("white")
sns.boxplot(data=combined_dfs, palette="viridis")
plt.xlabel("encoding/decoding method")
plt.ylabel("value")
plt.show()
