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


def parse_args():
    """Parse the command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--default-parameters",
        action="store_true",
        help="Enable the default parameters instead of quaternions",
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

    bilby.core.utils.setup_logger(
        outdir=outdir, label=label, log_level="WARNING"
    )

    duration = 4.0
    sampling_frequency = 2048.0

    np.random.seed(151226)

    injection_parameters = dict(
        total_mass=66.0,
        mass_ratio=0.9,
        a_1=0.4,
        a_2=0.3,
        tilt_1=0.5,
        tilt_2=1.0,
        phi_12=1.7,
        phi_jl=0.3,
        luminosity_distance=2000,
        theta_jn=0.4,
        psi=2.659,
        phase=1.3,
        geocent_time=1126259642.413,
        ra=1.375,
        dec=-1.2108,
    )

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

    # Set up prior
    priors = bilby.gw.prior.BBHPriorDict()
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
    else:
        print("Not using the quaternions")

    fixed_parameters = ["phi_12", "phi_jl", "a_1", "a_2", "tilt_1", "tilt_2"]
    for fp in fixed_parameters:
        priors[fp] = injection_parameters[fp]

    likelihood = GravitationalWaveTransientWithQuaternions(
        interferometers=ifos,
        waveform_generator=waveform_generator,
        priors=priors,
        phase_marginalization=False,
        distance_marginalization=True,
    )

    reparameterisations = {"mass_ratio": "default"}

    result = bilby.core.sampler.run_sampler(
        likelihood=likelihood,
        priors=priors,
        outdir=outdir,
        injection_parameters=injection_parameters,
        label=label,
        conversion_function=generate_all_bbh_parameters,
        flow_class="GWFlowProposal",
        sampler="nessai",
        resume=False,
        plot=True,
        nlive=2000,
        seed=150914,
        flow_config=dict(model_config=dict(n_transforms=6)),
        n_pool=16,
        reparameterisations=reparameterisations,
        reset_flow=8,
    )

    result.plot_corner()


if __name__ == "__main__":
    main()
