import os
from pathlib import Path


# PLOTS_PATH = Path(__file__).resolve().parents[1] / "saved_results"
PLOTS_PATH = Path("saved_results")
r_script_path = os.path.join("plots_davide", "make_r_spherical_plots_ut.r")


def _get_path(name):
    return PLOTS_PATH / name / "signal_data"


def run_r_plots(file_name):
    folder, name = os.path.split(file_name)
    os.system(" ".join(["Rscript --vanilla", r_script_path, str(file_name), folder]))
    return


if __name__ == "__main__":
    run_r_plots(_get_path("5OAdecoding714_USAT"))
    run_r_plots(_get_path("5OAdecoding714_allrad_maxre"))
    #
    run_r_plots(_get_path("704transcoding5OA_USAT"))
    run_r_plots(_get_path("704transcoding5OA_direct"))
    #
    run_r_plots(_get_path("ex_decoding_301_irregular"))
    run_r_plots(_get_path("ex_decoding_301irregular_vbap.png"))
    #
    run_r_plots(_get_path("panning51_USAT"))
    run_r_plots(_get_path("panning51_direct"))
