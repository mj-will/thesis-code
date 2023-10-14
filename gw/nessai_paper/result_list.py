"""Get the list of result files"""
import glob
import json
import os

from natsort import natsorted

from thesis_utils.io import write_json

from reconstruct_dynesty_91 import RECON_FILE

# Results were moved to scratch
NESSAI_BASE_PATH = (
    "/scratch/michael.williams/projects/flowproposal-cbc-tests/old_paper_runs/"
)
DYNESTY_BASE_PATH = "/home/michael.williams/git_repos/nessai-paper/analysis/"

PATHS = dict(
    dynesty=dict(
        nomarg=f"{DYNESTY_BASE_PATH}/outdir_dynesty/result/dynesty",
        marg=f"{DYNESTY_BASE_PATH}/outdir_dynesty/result/distance_marginalisation",
    ),
    nessai=dict(
        nomarg=f"{NESSAI_BASE_PATH}/outdir_nessai/result/reduced_overhead",
        marg=f"{NESSAI_BASE_PATH}/outdir_nessai/result/marg_dist_reduced_overhead",
    ),
)


def find_files(path: str, name: str) -> dict[str, str]:
    files = {}
    for i in range(128):
        if os.path.exists(
            (fn := f"{path}_data{i}_0_analysis_H1L1V1_{name}_result.json")
        ):
            files[str(i)] = fn
        else:
            files[str(i)] = None
    return files


def file_is_valid(filename: str) -> bool:
    valid = True
    valid &= os.path.exists(filename)
    valid &= os.path.getsize(filename) > 0
    return valid


def main() -> None:
    files = dict()

    for sampler, paths in PATHS.items():
        print(f"Finding files for {sampler}")
        files[sampler] = dict()
        for label, path in paths.items():
            print(f"Finding {label} files in {path}")
            data = find_files(path, sampler)
            n = sum([v is not None for v in data.values()])
            print(f"Found {n} result files")
            files[sampler][label] = data

    dynesty_missing_nomarg = [
        k for k, v in files["dynesty"]["nomarg"].items() if v is None
    ]
    dynesty_missing_marg = [
        k for k, v in files["dynesty"]["marg"].items() if v is None
    ]
    # Add missing dynesty results
    print(f"Missing dynesty results (no marg): {dynesty_missing_nomarg}")
    print(f"Missing dynesty results (marg): {dynesty_missing_marg}")

    files["dynesty"]["nomarg"][
        "77"
    ] = f"{DYNESTY_BASE_PATH}/outdir_dynesty/result/rerun_sampler_seed_data1_0_analysis_H1L1V1_dynesty_result.json"
    files["dynesty"]["nomarg"][
        "88"
    ] = f"{DYNESTY_BASE_PATH}/outdir_dynesty/result/rerun_sampler_seed_data2_0_analysis_H1L1V1_dynesty_result.json"

    files["dynesty"]["marg"][
        "88"
    ] = f"{DYNESTY_BASE_PATH}/outdir_dynesty/result/rerun_second_sampler_seed_distance_marg_data0_0_analysis_H1L1V1_dynesty_result.json"
    files["dynesty"]["marg"][
        "109"
    ] = f"{DYNESTY_BASE_PATH}/outdir_dynesty/result/rerun_second_sampler_seed_distance_marg_data1_0_analysis_H1L1V1_dynesty_result.json"
    files["dynesty"]["marg"][
        "115"
    ] = f"{DYNESTY_BASE_PATH}/outdir_dynesty/result/rerun_sampler_seed_distance_marg_data2_0_analysis_H1L1V1_dynesty_result.json"

    print("Checking files")
    for sampler, data in files.items():
        for label, paths in data.items():
            missing = [k for k, v in paths.items() if not file_is_valid(v)]
            if missing:
                print(f"{sampler} {label} is missing: {missing}")
            else:
                print(f"All files are valid for {sampler} {label}")

    # Replace corrupted file with reconstructed version
    files["dynesty"]["nomarg"]["91"] = RECON_FILE

    write_json(files, "result_files_lookup.json")


if __name__ == "__main__":
    main()
