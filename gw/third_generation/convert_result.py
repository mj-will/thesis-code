#!/usr/bin/env python
"""Convert a bilby result file from JSON to HDF5"""
import os
import sys
import bilby
import h5py
import numpy as np


def main(result_file, outdir, filename):
    result = bilby.core.result.read_in_result(result_file)
    os.makedirs(outdir, exist_ok=True)

    records = result.posterior[result.search_parameter_keys].to_records(
        index=False
    )
    post = np.array(records, dtype=records.dtype.descr)

    print(post)

    with h5py.File(os.path.join(outdir, filename), "w") as f:
        # post = result.posterior[result.search_parameter_keys].to_numpy()
        f.create_dataset("posterior_samples", data=post)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise RuntimeError(
            "Usage: convert_result.py <result file> <outdir> <filename>"
        )

    main(sys.argv[1], sys.argv[2], sys.argv[3])
