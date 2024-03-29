#!/usr/bin/env python
"""
Sampling with quaterions
"""
import argparse

import bilby
import numpy as np

from quaternions import (
    GravitationalWaveTransientWithQuaternions,
    convert_with_quaternions,
    generate_all_bbh_parameters,
)

from thesis_utils.gw import injections


def parse_args():
    """Parse the command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--default-parameters",
        action="store_true",
        help="Enable the default parameters instead of quaternions",
    )
    parser.add_argument(
        "--dynesty",
        action="store_true",
        help="Sample with dynesty",
    )
    parser.add_argument(
        "--label", type=str, help="Label for the run", default=None
    )
    return parser.parse_args()


def main():

    args = parse_args()

    outdir = "./outdir/"

    if args.label is not None:
        label = args.label
    elif args.default_parameters:
        label = "default"
    else:
        label = "quaternions"

    print(f"Results will be saved to {outdir}")

    bilby.core.utils.setup_logger(
        outdir=outdir,
        label=label,
        log_level="INFO",
    )

    duration = 4.0
    sampling_frequency = 2048.0

    np.random.seed(151226)

    injection_parameters = injections.BBH_GW150914.bilby_format()

    print("Creating waveform generator")
    waveform_arguments = dict(
        waveform_approximant="IMRPhenomPv2", reference_frequency=50.0
    )

    waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
        sampling_frequency=sampling_frequency,
        duration=duration,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=(convert_with_quaternions),
        waveform_arguments=waveform_arguments,
    )

    # Set up interferometers
    ifos = bilby.gw.detector.InterferometerList(["H1", "L1", "V1"])
    ifos.set_strain_data_from_power_spectral_densities(
        sampling_frequency=sampling_frequency,
        duration=duration,
        start_time=injection_parameters["geocent_time"] - 3,
    )
    ifos.inject_signal(
        waveform_generator=waveform_generator, parameters=injection_parameters
    )

    print("Creating priors")
    priors = bilby.gw.prior.BBHPriorDict()
    priors["chirp_mass"] = bilby.gw.prior.UniformInComponentsChirpMass(20, 40)
    priors["geocent_time"] = bilby.core.prior.Uniform(
        minimum=injection_parameters["geocent_time"] - 0.1,
        maximum=injection_parameters["geocent_time"] + 0.1,
        name="geocent_time",
        latex_label="$t_c$",
        unit="$s$",
    )

    if not all([p in priors.keys() for p in ["phase", "theta_jn", "psi"]]):
        raise RuntimeError("Missing source any parameters")

    if not args.default_parameters:
        print("Using the quaternions")
        priors.pop("phase")
        priors.pop("theta_jn")
        priors.pop("psi")
        # Add the priors
        for i in range(4):
            priors[f"q_{i}"] = bilby.core.prior.Gaussian(
                0, 1, name=f"q_{i}", latex_label=f"$q_{i}$"
            )
        # priors["q_0"] = bilby.core.prior.HalfNormal(1, label="$q_0$")
        # priors["q_3"] = bilby.core.prior.HalfNormal(1, label="$q_3$")
    else:
        print("Not using the quaternions")

    fixed_parameters = ["phi_12", "phi_jl", "a_1", "a_2", "tilt_1", "tilt_2"]
    for fp in fixed_parameters:
        priors[fp] = injection_parameters[fp]

    print("Creating quaternion likelihood")

    likelihood = GravitationalWaveTransientWithQuaternions(
        interferometers=ifos,
        waveform_generator=waveform_generator,
        priors=priors,
        phase_marginalization=False,
        distance_marginalization=True,
        time_marginalization=False,
    )

    print("Starting sampling")

    if args.dynesty:
        sampler = "dynesty"
        kwargs = dict(
            nlive=1000,
            sample="acceptance-walk",
            naccept=20,
            check_point_plot=True,
            check_point_delta_t=1800,
            print_method="interval-60",
        )
    else:
        sampler = "nessai"
        kwargs = dict(
            nlive=2000,
            flow_class="FlowProposal",
            flow_config=dict(model_config=dict(n_blocks=6, n_neurons=64)),
            n_pool=8,
            # reparameterisations=reparameterisations,
            use_default_reparameterisations=True,
            fallback_reparameterisation="z-score",
            reset_flow=8,
        )

    result = bilby.core.sampler.run_sampler(
        likelihood=likelihood,
        priors=priors,
        outdir=outdir,
        injection_parameters=injection_parameters,
        label=label,
        conversion_function=generate_all_bbh_parameters,
        sampler=sampler,
        resume=False,
        plot=True,
        seed=150914,
        **kwargs,
    )

    result.plot_corner()


if __name__ == "__main__":
    main()
