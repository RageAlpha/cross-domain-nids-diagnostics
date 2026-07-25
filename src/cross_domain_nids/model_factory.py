"""Construct released estimators directly from auditable JSON parameters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sklearn.ensemble import RandomForestClassifier

from cross_domain_nids.models.pipeline_models import (
    LabelEncodedXGBClassifier,
    TorchMLPClassifier,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARAMETERS_PATH = REPOSITORY_ROOT / "data" / "model_hyperparameters.json"


def load_parameter_record(
    experiment: str,
    model: str,
    parameters_path: Path | str = DEFAULT_PARAMETERS_PATH,
) -> dict[str, Any]:
    """Return the single released parameter record for an experiment/model."""
    path = Path(parameters_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    matches = [
        record
        for record in payload["templates"]
        if record["experiment"] == experiment and record["model"] == model
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one parameter record for {experiment}/{model}; "
            f"found {len(matches)} in {path}"
        )
    return matches[0]


def construct_estimator(
    experiment: str,
    model: str,
    seed: int,
    parameters_path: Path | str = DEFAULT_PARAMETERS_PATH,
):
    """Construct the historical estimator class with released parameters.

    The only deliberate per-run changes are ``random_state=seed`` and
    ``verbose=False``, matching the historical template-cloning runners.
    """
    record = load_parameter_record(experiment, model, parameters_path)
    parameters = dict(record["parameters"])
    parameters["random_state"] = int(seed)
    if "verbose" in parameters:
        parameters["verbose"] = False
    if model == "MLP" and isinstance(parameters.get("hidden_layer_sizes"), list):
        parameters["hidden_layer_sizes"] = tuple(parameters["hidden_layer_sizes"])

    constructors = {
        "RandomForest": RandomForestClassifier,
        "XGBoost": LabelEncodedXGBClassifier,
        "MLP": TorchMLPClassifier,
    }
    try:
        constructor = constructors[model]
    except KeyError as exc:
        raise ValueError(f"Unsupported released model: {model}") from exc
    return constructor(**parameters)


def constructor_signature(
    experiment: str,
    model: str,
    seed: int,
    parameters_path: Path | str = DEFAULT_PARAMETERS_PATH,
) -> dict[str, Any]:
    """Return normalized constructor metadata for verification and manifests."""
    estimator = construct_estimator(experiment, model, seed, parameters_path)
    parameters = estimator.get_params(deep=False)
    if isinstance(parameters.get("hidden_layer_sizes"), tuple):
        parameters["hidden_layer_sizes"] = list(parameters["hidden_layer_sizes"])
    return {
        "experiment": experiment,
        "model": model,
        "class": f"{type(estimator).__module__}.{type(estimator).__name__}",
        "parameters": parameters,
    }
