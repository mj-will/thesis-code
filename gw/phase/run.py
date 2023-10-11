#!/usr/bin/env python
"""
Sampling phase with default settings
"""
import bilby
import numpy as np

from thesis_utils.gw import injections


def main():

    outdir = "./outdir/"
    label = "nessai_no_gw"

    bilby.core.utils.setup_logger(
        outdir=outdir,
        label=label,
        log_level="INFO",
    )

    duration = 4.0
    sampling_frequency = 2048.0

    np.random.seed(150914)

    injection_parameters = injections.BBH_GW150914.bilby_format()

    waveform_arguments = dict(
        waveform_approximant="IMRPhenomPv2", reference_frequency=50.0
    )

    waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
        sampling_frequency=sampling_frequency,
        duration=duration,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        waveform_arguments=waveform_arguments,
    )

    print("Constructing IFOs")
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
    print("Constructing priors")
    priors = bilby.gw.prior.BBHPriorDict()
    priors["chirp_mass"] = bilby.gw.prior.UniformInComponentsChirpMass(20, 40)
    priors["geocent_time"] = bilby.core.prior.Uniform(
        minimum=injection_parameters["geocent_time"] - 0.1,
        maximum=injection_parameters["geocent_time"] + 0.1,
        name="geocent_time",
        latex_label="$t_c$",
        unit="$s$",
    )

    print("Constructing likelihood")

    likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
        interferometers=ifos,
        waveform_generator=waveform_generator,
        priors=priors,
        phase_marginalization=False,
        distance_marginalization=True,
    )

    print("Starting sampling")

    reparameterisations = {
        "mass_ratio": "mass",
        # "chirp_mass": "z",
        # "a_1": "z-score",
        # "a_2": "z-score",
        "psi": "angle-pi",
        "theta_jn": "angle-sine",
        # "psi": "angle-pi",
        # "theta_jn": "angle-sine",
        "phase": "angle-2pi",
        # "alpha-beta": {"parameters": ["phase", "psi"]}
    }

    result = bilby.core.sampler.run_sampler(
        likelihood=likelihood,
        priors=priors,
        outdir=outdir,
        injection_parameters=injection_parameters,
        label=label,
        conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
        flow_class="FlowProposal",
        sampler="nessai",
        resume=True,
        plot=True,
        nlive=2000,
        seed=150914,
        flow_config=dict(model_config=dict(n_blocks=10, n_neurons=128)),
        reset_flow=8,
        n_pool=8,
        volume_fraction=0.98,
        use_default_reparameterisations=True,
        fallback_reparameterisation="z-score",
        # reparameterisations=reparameterisations,
        # reverse_reparameterisations=True,
        # pytorch_threads=8,
    )

    result.plot_corner()


if __name__ == "__main__":
    main()
