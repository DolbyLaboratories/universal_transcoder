import os
from pathlib import Path


# PLOTS_PATH = Path(__file__).resolve().parents[1] / "saved_results"
PLOTS_PATH = Path("saved_results")
R_SCRIPT_PATH = os.path.join("plots_paper", "make_r_spherical_plots")
PDF_CROP_PATH = os.path.join(os.sep, "Users", "dscai", "src", "pdfcrop", "pdfcrop.pl")


def _get_path(name):
    return PLOTS_PATH / name / "signal_data"


def run_r_plots(file_name):
    txt_path = _get_path(file_name)
    folder, _ = os.path.split(txt_path)
    name = file_name.split("_")[0]
    # r_script = "_".join([R_SCRIPT_PATH, name]) + ".r"
    r_script = "_".join([R_SCRIPT_PATH, "all"]) + ".r"
    is_hemispherical = 0 if ("5OAdecoding714" in file_name) else 1
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
    run_r_plots("5OAdecoding714_USAT")
    run_r_plots("5OAdecoding714_allrad_maxre")
    #
    run_r_plots("704transcoding5OA_USAT")
    run_r_plots("704transcoding5OA_direct")
    #
    run_r_plots("ex_decoding_301_irregular")
    run_r_plots("ex_decoding_301irregular_vbap.png")
    #
