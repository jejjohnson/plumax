# Likelihoods

Observation-noise models for the Tier I likelihoods: `GaussianNoise` (the default, today's numbers), `StudentTNoise` for heavy-tailed retrievals near the detection floor, and `HeteroscedasticNoise` for a per-pixel retrieval-error map. Pass one as `noise=` to the plume and puff inference models; `log_prob` gives the same likelihood in plain JAX for a training loss. **Tier I.**

::: plumax.gauss_plume.likelihoods
