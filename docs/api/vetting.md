# Plume vetting

Plume-candidate vetting on arrays (L1): which of a candidate's pixels are in (the seeds) and which are out (the background pool), and how this scene's clutter scores in the candidate's shape (the null), on radtrax's plume-agnostic absorption-shape test (`radtrax.retrieval.absorption`: the pairing, the ratio spectrum, the fit and `D_norm`; see radtrax's [absorption-shape test](https://jejjohnson.github.io/radtrax/absorption-test/) page).

Errors are radtrax's public exception classes (`ParameterError`, `DegenerateInputError`, `ShapeError`), each also a `ValueError`.

::: plumax.vetting
    options:
      members: false

## `seeds` — the seeds and the background pool

::: plumax.vetting
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - select_seeds
        - background_pool

## `null` — a null from the same scene

::: plumax.vetting
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - null_masks
        - null_pvalue
