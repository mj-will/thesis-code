#!/usr/bin/env python
"""
Run nessai with bilby for GW150914-like injection
"""
import argparse

import bilby
import numpy as np

from thesis_utils.gw import injections, quaternions
from thesis_utils.gw.utils import PRECESSING_SPIN_PARAMETERS
from thesis_utils.io import load_json


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--outdir", type=str, default="outdir/")
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=150914)
    parser.add_argument("--n-pool", type=int, default=None)
    parser.add_argument("--two-detector", action="store_true")
    parser.add_argument(
        "--fix-spins",
        action="store_true",
        help="Fix the spin parameters to their injected values.",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    outdir = args.outdir
    label = args.label

    config = load_json(args.config)

    bilby.core.utils.setup_logger(
        outdir=outdir,
        label=label,
        log_level="INFO",
    )

    duration = 4.0
    sampling_frequency = 2048.0

    np.random.seed(args.seed)

    injection_parameters = injections.BBH_GW150914.bilby_format()
    injection_parameters["luminosity_distance"] = 1000.0

    waveform_arguments = dict(
        waveform_approximant="IMRPhenomPv2", reference_frequency=50.0
    )

    waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
        sampling_frequency=sampling_frequency,
        duration=duration,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=quaternions.convert_with_quaternions,
        waveform_arguments=waveform_arguments,
    )

    if args.two_detector:
        ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
    else:
        ifos = bilby.gw.detector.InterferometerList(["H1", "L1", "V1"])

    ifos.set_strain_data_from_power_spectral_densities(
        sampling_frequency=sampling_frequency,
        duration=duration,
        start_time=injection_parameters["geocent_time"] - 3,
    )
    ifos.inject_signal(
        waveform_generator=waveform_generator, parameters=injection_parameters
    )

    priors = bilby.gw.prior.BBHPriorDict()
    priors["chirp_mass"] = bilby.gw.prior.UniformInComponentsChirpMass(20, 40)
    priors["geocent_time"] = bilby.core.prior.Uniform(
        minimum=injection_parameters["geocent_time"] - 0.1,
        maximum=injection_parameters["geocent_time"] + 0.1,
        name="geocent_time",
        latex_label="$t_c$",
        unit="$s$",
    )

    if config.get("quaternions", False):
        priors.pop("phase")
        priors.pop("theta_jn")
        priors.pop("psi")
        for i in range(4):
            priors[f"q_{i}"] = bilby.core.prior.Gaussian(
                0, 1, name=f"q_{i}", latex_label=f"$q_{i}$"
            )
    elif config.get("delta_phase", False):
        priors.pop("phase")
        priors["delta_phase"] = bilby.core.prior.Uniform(
            0, 2 * np.pi, name="delta_phase", latex_label="$\Delta\phi_c$"
        )

    if args.fix_spins:
        for p in PRECESSING_SPIN_PARAMETERS:
            priors[p] = injection_parameters[p]

    likelihood = quaternions.GravitationalWaveTransientWithQuaternions(
        interferometers=ifos,
        waveform_generator=waveform_generator,
        priors=priors,
        phase_marginalization=config["phase_marginalization"],
        distance_marginalization=True,
        time_marginalization=False,
    )

    result = bilby.core.sampler.run_sampler(
        likelihood=likelihood,
        priors=priors,
        outdir=outdir,
        injection_parameters=injection_parameters,
        label=label,
        conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
        sampler="nessai",
        resume=True,
        plot=True,
        seed=args.seed,
        n_pool=args.n_pool,
        **config["sampler_kwargs"],
    )

    result.plot_corner()


if __name__ == "__main__":
    main()
