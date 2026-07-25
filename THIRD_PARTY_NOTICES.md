# Third-party notices

No third-party source tree or externally licensed implementation is vendored
in this release. The DANN/gradient-reversal, NetFlow preprocessing,
calibration, MMD, model-wrapper, and experiment-runner code in `src/` and
`experiments/` is the repository's implementation for the accompanying
manuscript.

The software depends on the packages pinned in `requirements.txt` and
`environment.yml`. Those packages remain subject to their own licenses; the
repository MIT license does not replace or relicense them.

The four NF-UQ-NIDS-v3 datasets are external provider data. They are not
redistributed. Users must obtain them from
https://staff.itee.uq.edu.au/marius/NIDS_datasets/ under the provider's terms.
The expected filenames, provider DOIs, and SHA-256 values are recorded in
`data/README.md` and `data/dataset_checksums.csv`.

If code is later copied from another repository, its source, original license,
and local modifications must be documented here before release.
