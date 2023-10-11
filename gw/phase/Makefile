.PHONY : reparam_comparison
no_gw_comparison:
	python plot_comparison.py \
		--results outdir/paper_default_phase_marg_result.json outdir/no_gw_v1_result.json \
		--parameters  chirp_mass mass_ratio a_1 a_2 ra dec theta_jn phase psi \
		--filename "phase_comparison_no_gw" \
		--labels "With phase marg." "Without phase marg. + No GW"

reparam_comparison :
	python plot_comparison.py \
		--results outdir/paper_default_phase_marg_result.json outdir/gw_v18_result.json \
		--parameters  chirp_mass mass_ratio a_1 a_2 ra dec theta_jn phase psi \
		--filename "phase_comparison_reparam" \
		--labels "With phase marg." "Without phase marg."
