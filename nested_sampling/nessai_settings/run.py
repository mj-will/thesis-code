# /usr/bin/env python
import argparse
import os

from thesis_utils.nessai import run_nessai
from thesis_utils.io import load_json
from thesis_utils.random import seed_everything


def parse_args() -> argparse.ArgumentParser:
    args = argparse.ArgumentParser()
    args.add_argument("--config", type=str, required=True)
    args.add_argument("--model", type=str, default=None)
    args.add_argument("--dims", type=int, default=None)
    args.add_argument("--model-config", type=str, default=None)
    args.add_argument("--output", type=str, default="outdir")
    return args.parse_args()


def main():

    args = parse_args()
    config = load_json(args.config)

    if "seed" not in config:
        config["seed"] = 1234
    seed_everything(config["seed"])

    if args.model_config is not None:
        model_config = load_json(args.model_config)
    elif args.model is None and args.dims is None:
        raise RuntimeError(
            "Must specify model and dims when not specifying model_config"
        )
    else:
        model_config = dict(name=args.model, dims=args.dims)

    name = os.path.splitext(os.path.basename(args.config))[0]
    print(name)
    if args.model_config:
        model_config_name = os.path.splitext(
            os.path.basename(args.model_config)
        )[0]
        output = os.path.join(args.output, f"{name}_{model_config_name}")
    else:
        output = os.path.join(args.output, f"{name}_{args.model}_{args.dims}d")

    if os.path.exists(os.path.join(output, "result.hdf5")):
        raise RuntimeError(f"Run already exists in {output}")

    run_nessai(output=output, model_config=model_config, **config)


if __name__ == "__main__":
    main()
