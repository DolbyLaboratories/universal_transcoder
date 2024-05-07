import os
from pathlib import Path


PLOTS_PATH = Path("saved_results")
R_SCRIPT_PATH = os.path.join("paper", "plots_paper", "make_r_spherical_plots")
PDF_CROP_PATH = os.path.join(os.sep, "Users", "dscai", "src", "pdfcrop", "pdfcrop.pl")


def _get_path(name):
    return PLOTS_PATH / name / "signal_data"


def run_r_plots(file_name):
    txt_path = _get_path(file_name)
    folder, _ = os.path.split(txt_path)
    name = file_name.split("_")[0]
    # r_script = "_".join([R_SCRIPT_PATH, name]) + ".r"
    r_script = "_".join([R_SCRIPT_PATH, "all"]) + ".r"
    is_hemispherical = 0 if ("ex1_50Ato704" in file_name) else 1
    is_714 = 1 if ("7" in file_name) else 0

    # make plots
    os.system(
        " ".join(
            [
                "Rscript --vanilla",
                r_script,
                str(txt_path),
                folder,
                str(is_hemispherical),
                str(is_714),
            ]
        )
    )

    # shrink pdfs
    pdfs = [file for file in os.listdir(folder) if os.path.splitext(file)[-1] == ".pdf"]
    for pdf in pdfs:
        file_path = os.path.join(folder, pdf)
        os.system(" ".join([PDF_CROP_PATH, file_path, file_path]))
    return


if __name__ == "__main__":
    run_r_plots("ex1_50Ato704_USAT")
    run_r_plots("ex1_50Ato704_ALLRAD_maxre")
    #
    run_r_plots("ex2_704to5OA_USAT")
    run_r_plots("ex2_704to5OA_direct")
    #
    run_r_plots("ex3_50to301irr_USAT")
    run_r_plots("ex3_50to301irr_vbap")
    #
