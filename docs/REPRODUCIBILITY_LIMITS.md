# Reproducibility scope

Quick verification checks numerical reference evidence, frozen inputs and release
hashes. Synthetic tests exercise training, paired controls, source validation,
selection isolation, interrupted recovery and rejection of incomplete evidence.
These checks are distinct from full retraining.

Full revision execution requires provider CSVs, the pinned environment and CUDA.
H3 adapts to a fixed 5% target sample and evaluates the entire target file,
including adaptation rows. The readout is target-supervised and post-hoc.

Returned execution-review tables supply the new compact references. Their
arithmetic is independently checked by `experiments/revision/evidence.py`.
Original full result trees, checkpoints, prediction streams and current selection
locks are not included; fresh execution generates them. Archived review statuses
are provenance, not a new validation of absent checkpoints. This packaging task
did not rerun the full experiments.

Historical H2/H3 summaries, locks and commands retain the original 20-mask protocol.
They must not be reported as corrected 10-mask results. The sensitivity runner
requires historical execution products as well as the new stages. The historical
gate grid can also be checked directly from compact deltas without training.

Historical H3 complement recalculation needs regenerated or separately deposited
predictions. Fresh calibration inference needs fitted H1 models. Historical
source-class confusion is contextual; its original fitted models and predictions
are not included. Supplementary historical plots can be regenerated from the compact count and
shift tables; their visual styling is not claimed to be identical to the manuscript.

Dependencies pin direct versions, not all transitive packages, GPU drivers or
CUDA builds. Formal revision experiments ran on a separate host. Local checks do
not establish cross-hardware bitwise equivalence. Figure numbers are validated;
PDF bytes and typography can vary by renderer.

Release changes reorganize imports and paths, extract helpers, construct models
from matching parameter dictionaries, refit source preprocessing on frozen rows,
translate public text and create fresh run manifests. Archived hashes are not
rewritten to claim that the execution code is unchanged.
