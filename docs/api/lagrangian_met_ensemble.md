# Met ensemble

One footprint per meteorological ensemble member (`ensemble_footprints`, on `GriddedWind`s or uniform winds), and the Gaussian inversion that marginalises over them (`ensemble_inversion`): the mean footprint, with the members' spread at the prior mean added to the observation covariance as a transport error (`transport_covariance`). Needs the `assim` extra for the inversion. **Tier II.**

::: plumax.lagrangian.met_ensemble
