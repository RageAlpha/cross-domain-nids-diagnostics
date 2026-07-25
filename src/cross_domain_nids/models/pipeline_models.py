"""
Model helpers for end-to-end sklearn Pipelines.

The baseline pipeline stores preprocessing and the classifier in one artifact.
XGBoost needs label encoding internally, so this module provides a small
sklearn-compatible wrapper that accepts string labels and returns string labels.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from cross_domain_nids.config import SEED


class LabelEncodedXGBClassifier(BaseEstimator, ClassifierMixin):
    """XGBClassifier wrapper that handles string class labels internally."""

    def __init__(
        self,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=1.0,
        colsample_bytree=1.0,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        device=None,
        random_state=SEED,
        n_jobs=-1,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.objective = objective
        self.eval_metric = eval_metric
        self.tree_method = tree_method
        self.device = device
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _xgb_params(self):
        params = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "objective": self.objective,
            "eval_metric": self.eval_metric,
            "tree_method": self.tree_method,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
        }
        if self.device is not None:
            params["device"] = self.device
        return params

    def fit(self, X, y):
        self.label_encoder_ = LabelEncoder()
        y_enc = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_

        self.model_ = XGBClassifier(**self._xgb_params())
        self.model_.fit(X, y_enc)
        return self

    def predict(self, X):
        y_enc = self.model_.predict(X)
        return self.label_encoder_.inverse_transform(np.asarray(y_enc, dtype=int))

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    """CUDA-accelerated MLP with a sklearn-compatible classifier interface."""

    def __init__(
        self,
        hidden_layer_sizes=(128, 64),
        alpha=0.0001,
        learning_rate=0.001,
        batch_size=4096,
        max_epochs=30,
        patience=5,
        validation_fraction=0.1,
        require_gpu=True,
        random_state=SEED,
        verbose=False,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.require_gpu = require_gpu
        self.random_state = random_state
        self.verbose = verbose

    def _check_torch(self):
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "TorchMLPClassifier requires PyTorch. Install a CUDA-enabled "
                "torch build before training the MLP baseline."
            ) from exc

        if self.require_gpu and not torch.cuda.is_available():
            raise RuntimeError(
                "MLP baseline requires CUDA GPU acceleration, but torch.cuda "
                "is not available. Use a CUDA-enabled PyTorch environment."
            )

        return torch

    def _build_model(self, torch, n_features, n_classes):
        layers = []
        in_features = n_features
        for hidden_size in self.hidden_layer_sizes:
            layers.append(torch.nn.Linear(in_features, hidden_size))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.BatchNorm1d(hidden_size))
            layers.append(torch.nn.Dropout(p=0.1))
            in_features = hidden_size
        layers.append(torch.nn.Linear(in_features, n_classes))
        return torch.nn.Sequential(*layers)

    def fit(self, X, y, validation_data=None):
        torch = self._check_torch()
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_ = str(device)

        X_np = np.asarray(X, dtype=np.float32)
        self.label_encoder_ = LabelEncoder()
        y_np = self.label_encoder_.fit_transform(y).astype(np.int64)
        self.classes_ = self.label_encoder_.classes_

        n_samples = X_np.shape[0]
        if validation_data is None:
            rng = np.random.default_rng(self.random_state)
            indices = rng.permutation(n_samples)
            n_val = max(1, int(n_samples * self.validation_fraction))
            val_idx = indices[:n_val]
            train_idx = indices[n_val:]

            X_train_np = X_np[train_idx]
            y_train_np = y_np[train_idx]
            X_val_np = X_np[val_idx]
            y_val_np = y_np[val_idx]
            self.validation_source_ = "internal_validation_fraction"
        else:
            X_val_raw, y_val_raw = validation_data
            X_train_np = X_np
            y_train_np = y_np
            X_val_np = np.asarray(X_val_raw, dtype=np.float32)
            y_val_np = self.label_encoder_.transform(y_val_raw).astype(np.int64)
            self.validation_source_ = "explicit_validation_data"

        X_train = torch.from_numpy(X_train_np)
        y_train = torch.from_numpy(y_train_np)
        X_val = torch.from_numpy(X_val_np).to(device)
        y_val = torch.from_numpy(y_val_np).to(device)

        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        generator = torch.Generator()
        generator.manual_seed(self.random_state)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
            pin_memory=torch.cuda.is_available(),
        )

        self.model_ = self._build_model(
            torch,
            n_features=X_np.shape[1],
            n_classes=len(self.classes_),
        ).to(device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.alpha,
        )

        best_state = None
        best_val_loss = float("inf")
        stale_epochs = 0

        for epoch in range(1, self.max_epochs + 1):
            self.model_.train()
            total_loss = 0.0
            for xb, yb in loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = self.model_(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().cpu()) * len(xb)

            self.model_.eval()
            with torch.no_grad():
                val_logits = self.model_(X_val)
                val_loss = float(criterion(val_logits, y_val).detach().cpu())

            if self.verbose:
                train_loss = total_loss / max(len(y_train_np), 1)
                print(
                    f"TorchMLP epoch {epoch:03d} | "
                    f"train_loss={train_loss:.5f} | val_loss={val_loss:.5f}"
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in self.model_.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.patience:
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.best_val_loss_ = best_val_loss
        self.n_epochs_ = epoch
        self.model_.eval()
        return self

    def _predict_logits(self, X):
        torch = self._check_torch()
        X_np = np.asarray(X, dtype=np.float32)
        device = torch.device(self.device_)
        outputs = []
        with torch.no_grad():
            for start in range(0, len(X_np), self.batch_size):
                batch = torch.from_numpy(X_np[start:start + self.batch_size]).to(device)
                outputs.append(self.model_(batch).detach().cpu().numpy())
        return np.vstack(outputs)

    def predict_proba(self, X):
        logits = self._predict_logits(X)
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)

    def predict(self, X):
        pred_idx = np.argmax(self.predict_proba(X), axis=1)
        return self.label_encoder_.inverse_transform(pred_idx)
