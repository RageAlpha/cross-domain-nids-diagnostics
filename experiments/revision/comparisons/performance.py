"""Bounded batch preparation and deferred audits, without changing sampling/math."""
import torch


class BatchedTensorDataset(torch.utils.data.TensorDataset):
    def __getitems__(self, indices):
        # Same indices from the original DataLoader RandomSampler; no new RNG.
        index = torch.as_tensor(indices, dtype=torch.long)
        return tuple(tensor.index_select(0, index) for tensor in self.tensors)


def identity_batch(batch):
    return batch


def make_loader(dataset, *, batch_size, generator, cuda=False):
    return torch.utils.data.DataLoader(
        BatchedTensorDataset(*dataset.tensors), batch_size=batch_size,
        shuffle=True, generator=generator, drop_last=False,
        collate_fn=identity_batch, pin_memory=cuda, num_workers=0,
    )


class FrozenBNAudit:
    """Check every target forward on device; raise before logging/checkpointing."""
    def __init__(self, model, device):
        self.buffers = tuple(
            value for layer in model.modules() if isinstance(layer, torch.nn.BatchNorm1d)
            for value in (layer.running_mean, layer.running_var, layer.num_batches_tracked)
        )
        self.valid = torch.ones((), dtype=torch.bool, device=device)

    def snapshot(self):
        return tuple(value.detach().clone() for value in self.buffers)

    def observe(self, before):
        for saved, current in zip(before, self.buffers):
            self.valid.logical_and_(torch.all(saved == current))

    def verify(self):
        if self.buffers and not bool(self.valid.item()):
            raise ValueError('Target forward changed frozen BatchNorm buffers')


def weighted_scalar_sum(values):
    """One device transfer, same Python float accumulation order as before."""
    if not values:
        return 0.0
    scalars = torch.stack([value for value, _ in values]).cpu().tolist()
    total = 0.0
    for value, (_, count) in zip(scalars, values):
        total += float(value) * count
    return total
