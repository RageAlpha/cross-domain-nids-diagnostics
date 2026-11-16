"""Frozen alignment losses, mRMR ranking and revision model implementations."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.stats import rankdata
from sklearn.preprocessing import LabelEncoder
from cross_domain_nids.models.pipeline_models import TorchMLPClassifier
from cross_domain_nids.models.dann import GradientReversal
from cross_domain_nids.vendor.rtdl_revisiting_models import FTTransformer
from experiments.h3.run_paired_domain_loss import save_training_checkpoint, _optimizer_to_device

def mi(a,b):
    _,a=np.unique(a,return_inverse=True);_,b=np.unique(b,return_inverse=True)
    n=int(b.max())+1
    joint=np.bincount(a*n+b,minlength=(int(a.max())+1)*n).reshape(-1,n).astype(float)
    joint/=joint.sum();prod=joint.sum(1)[:,None]*joint.sum(0)[None,:]
    nz=joint>0
    return float(np.sum(joint[nz]*np.log(joint[nz]/prod[nz])))


def rank_mrmr(x,y,names):
    # Empirical discrete MI, numeric columns binned by source empirical ranks.
    # Categories retain their encoded states. Constant features are placed last.
    d=np.zeros_like(x,dtype=np.int32)
    for j,name in enumerate(names):
        if name in ['PROTOCOL','L7_PROTO']:
            d[:,j]=np.unique(x[:,j],return_inverse=True)[1]
        else:
            # Mean ranks keep tied values together without collapsing rare
            # nonzero values into a repeated zero quantile boundary.
            d[:,j]=np.minimum(15,((rankdata(x[:,j],method='average')-.5)*16/len(x)).astype(int))
    rel=np.array([mi(d[:,j],y) for j in range(x.shape[1])])
    red=np.zeros((len(names),len(names)))
    for i in range(len(names)):
        for j in range(i,len(names)):
            red[i,j]=red[j,i]=mi(d[:,i],d[:,j])
    active=[j for j in range(len(names)) if np.unique(d[:,j]).size>1]
    constant=[j for j in range(len(names)) if j not in active]
    chosen=[]
    while active:
        scores=[rel[j]-(red[j,chosen].mean() if chosen else 0.) for j in active]
        winner=active[int(np.argmax(scores))]
        chosen.append(winner);active.remove(winner)
    return chosen+constant,rel,constant

def coral(s, t):
    if min(len(s), len(t)) < 2:
        raise ValueError('Covariance requires at least two rows per domain')
    a, b = s - s.mean(0), t - t.mean(0)
    cs, ct = a.T @ a / (len(s)-1), b.T @ b / (len(t)-1)
    return (cs-ct).square().sum() / (4*s.shape[1]**2)


def mmd(s, t, bandwidth):
    # bandwidth is sigma squared, fixed before each matched short run.
    def kernel_mean(a, b):
        distance = (a.square().sum(1)[:, None] + b.square().sum(1)[None, :] - 2*a@b.T).clamp_min(0)
        return sum(torch.exp(-distance/(2*bandwidth*factor)).mean() for factor in [.5, 1., 2.])/3
    return kernel_mean(s, s)+kernel_mean(t, t)-2*kernel_mean(s, t)


def numerical_tests():
    torch.manual_seed(7)
    s = torch.randn(8, 4, dtype=torch.float64, requires_grad=True)
    t = torch.randn(7, 4, dtype=torch.float64, requires_grad=True)
    checks = {}
    for name, fn in [('coral', coral), ('mmd', lambda a,b: mmd(a,b,2.))]:
        checks[name+'_gradcheck'] = bool(torch.autograd.gradcheck(fn, (s,t), atol=1e-5))
        checks[name+'_symmetric'] = bool(torch.allclose(fn(s,t),fn(t,s),atol=1e-10))
        checks[name+'_identity_zero'] = abs(float(fn(s,s).detach())) < 1e-10
    checks['coral_translation_invariant'] = abs(float(coral(s,s+3).detach())) < 1e-10
    checks['mmd_detects_translation'] = float(mmd(s,s+3,2.).detach()) > .1
    # Independent NumPy covariance and broadcasted-distance references.
    a,b=s.detach().numpy(),t.detach().numpy()
    cref=np.square(np.cov(a,rowvar=False)-np.cov(b,rowvar=False)).sum()/(4*a.shape[1]**2)
    def km(x,y):
        d=np.square(x[:,None,:]-y[None,:,:]).sum(2)
        return np.mean([np.exp(-d/(4*f)).mean() for f in [.5,1.,2.]])
    checks['coral_numpy_reference'] = bool(np.isclose(float(coral(s,t).detach()),cref))
    checks['mmd_numpy_reference'] = bool(np.isclose(float(mmd(s,t,2.).detach()),km(a,a)+km(b,b)-2*km(a,b)))
    assert all(checks.values()), checks
    return checks

class FTDomainModel(nn.Module):
    """Official default two-block FT, with shared normalized CLS representation."""
    def __init__(self):
        super().__init__()
        self.feature_extractor = FTTransformer(
            n_cont_features=49, cat_cardinalities=[], d_out=None,
            **FTTransformer.get_default_kwargs(n_blocks=2))
        self.output_norm = nn.LayerNorm(128)
        self.label_predictor = nn.Linear(128, 2)
        self.domain_classifier = nn.Sequential(
            GradientReversal(1.), nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward_label(self, x, return_features=False):
        z = torch.relu(self.output_norm(self.feature_extractor(x, None)))
        logits = self.label_predictor(z)
        return (logits, z) if return_features else logits

    def forward_domain(self, z):
        return self.domain_classifier(z)

    def optimizer(self):
        groups = self.feature_extractor.make_parameter_groups()
        groups += [dict(params=list(self.output_norm.parameters()), weight_decay=0.),
                   dict(params=[self.label_predictor.weight]),
                   dict(params=[self.label_predictor.bias], weight_decay=0.),
                   dict(params=[p for n, p in self.domain_classifier.named_parameters()
                                if not n.endswith('bias')]),
                   dict(params=[p for n, p in self.domain_classifier.named_parameters()
                                if n.endswith('bias')], weight_decay=0.)]
        return torch.optim.AdamW(groups, lr=1e-4, weight_decay=1e-5)

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
