"""
Floquet spectral diagnostics of the unitary rules (R32): quasienergies, half-chain
entanglement, level-spacing ratios, symmetry-resolved.

The heavy diagonalisation runs in Julia (HSF/generate_spectral_data.jl, see
`floquet.JULIA_SCRIPT`), writing results_spectra/{gate}/spec_*.jld2.  This package
reads those files, builds U independently from the exact engine for cross-checks,
and computes the statistics and figures.
"""
