"""MLP training with explicit source validation and epoch recovery."""
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from cross_domain_nids.models.pipeline_models import TorchMLPClassifier
from experiments.h3.run_paired_domain_loss import save_training_checkpoint, _optimizer_to_device

class RevisionMLP(TorchMLPClassifier):
    def fit(self, X, y, validation_data=None):
        if validation_data is None:
            raise ValueError('Corrected MLP requires explicit source validation data')
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

        checkpoint = Path(self.checkpoint_dir) / 'resume.pt'
        history, start_epoch = [], 1
        if checkpoint.exists():
            state = torch.load(checkpoint, map_location='cpu', weights_only=False)
            self.model_.load_state_dict(state['model'])
            optimizer.load_state_dict(state['optimizer'])
            _optimizer_to_device(optimizer, device)
            generator.set_state(state['loader_rng'])
            torch.set_rng_state(state['torch_rng'])
            if device.type == 'cuda':
                torch.cuda.set_rng_state_all(state['cuda_rng'])
            best_state, best_val_loss = state['best_state'], state['best_val_loss']
            stale_epochs, history = state['stale_epochs'], state['history']
            start_epoch = state['epoch'] + 1
        epoch = start_epoch - 1
        for epoch in range(start_epoch, self.max_epochs + 1):
            if stale_epochs >= self.patience:
                epoch -= 1
                break
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
            if not np.isfinite(val_loss):
                raise ValueError('Nonfinite validation loss')
            history.append({'epoch': epoch, 'train_loss': total_loss / len(y_train_np), 'source_val_loss': val_loss})
            save_training_checkpoint(checkpoint, epoch=epoch, model=self.model_.state_dict(),
                                     optimizer=optimizer.state_dict(), best_state=best_state,
                                     best_val_loss=best_val_loss, stale_epochs=stale_epochs, history=history,
                                     loader_rng=generator.get_state(), torch_rng=torch.get_rng_state(),
                                     cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None)
            print(f'MLP seed={self.random_state} epoch={epoch} val_loss={val_loss:.6f}', flush=True)
            if stale_epochs >= self.patience:
                break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.best_val_loss_ = best_val_loss
        self.n_epochs_ = epoch
        self.history_ = history
        self.model_.eval()
        return self
