# Data inputs

- `dataset_checksums.csv`: provider identifiers and SHA-256 checksums for the
  four raw NetFlow CSVs. Raw datasets are not included.
- `feature_schema.csv`: the 49-feature schema used by the manuscript.
- `model_hyperparameters.json`: recovered parameter dictionaries for the
  reported model configurations. H1 and H2 construct their estimators directly
  from this file.
- `reference_inputs/`: compact historical result tables required by the H1 and
  Oracle finalizers. These contain metrics only, not models or predictions.
- `splits/h1/`: the three fixed H1 source split maps and validation files.
- `splits/oracle/`: nine fixed target-oracle split maps plus their compact
  metadata, manifest, and validation checks.

All paths recorded in the Oracle manifest and metadata are repository-relative.
Dataset labels retain manuscript spelling, while split filenames use lowercase
slugs: `cicids2018`, `bot_iot`, and `ton_iot`.
