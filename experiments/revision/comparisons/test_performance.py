"""Sampling, RNG, loss logging and BN-audit invariants for the fast path."""
import unittest
import torch
from .performance import make_loader, FrozenBNAudit, weighted_scalar_sum


class PerformanceTests(unittest.TestCase):
    def test_batches_and_rng_identical(self):
        for n in (7, 4096, 10003):
            with self.subTest(rows=n):
                data = torch.utils.data.TensorDataset(torch.arange(n * 3).reshape(n, 3), torch.arange(n))
                old_rng = torch.Generator().manual_seed(2024)
                new_rng = torch.Generator().manual_seed(2024)
                old = torch.utils.data.DataLoader(data, batch_size=4096, shuffle=True, generator=old_rng)
                new = make_loader(data, batch_size=4096, generator=new_rng)
                for _ in range(3):
                    a, b = list(old), list(new)
                    self.assertEqual(len(a), len(b))
                    for left, right in zip(a, b):
                        for x, y in zip(left, right):
                            self.assertTrue(torch.equal(x, y))
                    self.assertTrue(torch.equal(old_rng.get_state(), new_rng.get_state()))

    def test_interrupted_iterator_rng_identical(self):
        data = torch.utils.data.TensorDataset(torch.arange(10003))
        states = []
        outputs = []
        for fast in (False, True):
            generator = torch.Generator().manual_seed(42)
            loader = (make_loader(data, batch_size=4096, generator=generator) if fast else
                      torch.utils.data.DataLoader(data, batch_size=4096, shuffle=True, generator=generator))
            iterator = iter(loader)
            next(iterator)
            # A partially consumed target iterator is discarded at the epoch boundary.
            outputs.append(next(iter(loader))[0])
            states.append(generator.get_state())
        self.assertTrue(torch.equal(*states))
        self.assertTrue(torch.equal(*outputs))

    def test_bn_audit_catches_transient_change(self):
        for device in (['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']):
            with self.subTest(device=device):
                model = torch.nn.Sequential(torch.nn.BatchNorm1d(4)).to(device)
                audit = FrozenBNAudit(model, device)
                before = audit.snapshot()
                audit.observe(before)
                audit.verify()
                model[0].running_mean.add_(1)
                audit.observe(before)
                model[0].running_mean.sub_(1)
                with self.assertRaises(ValueError):
                    audit.verify()

    def test_loss_log_accumulation_identical(self):
        for device in (['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']):
            with self.subTest(device=device):
                values = [(torch.tensor(v, device=device), n) for v, n in [(0.13,4096),(1.41,4096),(0.002,17)]]
                previous = 0.0
                for v, n in values:
                    previous += float(v.cpu()) * n
                self.assertEqual(weighted_scalar_sum(values), previous)


if __name__ == '__main__':
    unittest.main()
