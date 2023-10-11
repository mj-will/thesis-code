#!/usr/env/bin python
"""Python script for submitting runs via HTCondor"""
import argparse
import configparser
import logging
import itertools
import os
import sys

from pycondor import Job, Dagman


EXTRA_LINES = [
    "checkpoint_exit_code=130",
    "max_retries=5",
    "accounting_group=ligo.dev.o4.cbc.pe.bilby",
]


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, help="Output")
    parser.add_argument("--executable", "-e", help="Executable to run")
    parser.add_argument("--dims", type=int)
    parser.add_argument("--delay", type=int, nargs="+")
    parser.add_argument("--n-pool", type=int, nargs="+")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--name", type=str, default="nessai_scaling")
    return parser.parse_args()


def get_dag(output, args, n_pool, delay):
    """Get the DAG"""
    name = f"{args.name}"
    submit = os.path.join(output, "submit", "")
    dag = Dagman(name, submit=submit)
    log_path = os.path.join(output, "analysis_logs", "")
    os.makedirs(log_path, exist_ok=True)
    analysis_output = os.path.join(output, "analysis", "")

    tag = f"{delay}ms_n_pool_{n_pool}"

    job_name = f"{name}_{tag}"
    extra_lines = EXTRA_LINES + [
        f"log={log_path}{tag}.log",
        f"output={log_path}{tag}.out",
        f"error={log_path}{tag}.err",
    ]
    job = Job(
        name=job_name,
        executable=args.executable,
        queue=1,
        getenv=True,
        submit=submit,
        request_memory="2GB",
        request_cpus=n_pool,
        extra_lines=extra_lines,
    )
    job.add_arg(
        f"--dims={args.dims} --delay={delay} --n_pool={n_pool} "
        f"--seed={args.seed} --output={analysis_output}"
    )
    dag.add_job(job)
    return dag


def main():
    """Get and submit the job"""
    args = parse_args()
    base_output = os.path.abspath(args.output)
    os.makedirs(base_output, exist_ok=True)

    logging.info(
        f"Setting up runs for delay={args.delay}, n_pool={args.n_pool}"
    )
    output = os.path.join(base_output, f"delay_test_{args.dims}d")
    for delay, n_pool in itertools.product(args.delay, args.n_pool):
        dag = get_dag(output, args, delay=delay, n_pool=n_pool)
        dag.build_submit()


if __name__ == "__main__":
    main()
