import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from e3nn import o3
import numpy as np
import torch

from irrednet.coefficients import (
    BASIS_VERSION, MODEL_VERSION, LABEL_CONVENTION, reconstruct_tensors, rotate_tensor,
)
from irrednet.data import SpaceGroupDataset
from irrednet.losses import training_loss, dielectric_scalar_statistics
from irrednet.model import IrredNetSpaceGroup, SpaceGroupCoefficientHead, load_coefficient_model
from irrednet.train import make_model_args, evaluate
from .test_space_group import structures


class Decoder:
    def reconstruct(self, coefficients, batch):
        return reconstruct_tensors(coefficients, batch, "dielectric")


class TestDielectricEngineering(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cells = structures()
        cls.dataset = SpaceGroupDataset([
            {"structure": cells[name][0], "dielectric": np.diag([2., 3., 4.]), "id": name}
            for name in ("Pm-3m", "P1")], "dielectric")

    def batch(self):
        return self.dataset.collate([self.dataset[0], self.dataset[1]])

    def test_tensor_loss_differentiates_through_coefficients_and_keeps_label_residual(self):
        batch = self.batch()
        prediction = (batch.coefficient_labels + .3).requires_grad_()
        loss, terms = training_loss(Decoder(), prediction, batch, "dielectric", space="tensor")
        gradient, = torch.autograd.grad(loss, prediction)
        self.assertEqual(terms, 18)
        torch.testing.assert_close(gradient, 2 * (prediction - batch.coefficient_labels) / terms, atol=1e-6, rtol=1e-5)
        coefficient_loss, count = training_loss(Decoder(), prediction, batch, "dielectric")
        # The cubic label deliberately violates symmetry. Tensor loss must retain
        # its orthogonal residual; silently projecting the labels would hide it.
        self.assertGreater(float(loss.detach() * terms), float(coefficient_loss.detach() * count) + 1.)

    def test_tensor_huber_is_basis_invariant_but_coefficient_huber_is_not(self):
        torch.manual_seed(123)
        batch = self.batch()
        coefficients = batch.coefficient_labels + torch.arange(7.)
        changed = batch.clone()
        transformed = coefficients.clone()
        for start, stop in zip(batch.coefficient_ptr[:-1], batch.coefficient_ptr[1:]):
            gauge = torch.linalg.qr(torch.randn(int(stop-start), int(stop-start))).Q
            changed.cartesian_basis[start:stop] = gauge @ batch.cartesian_basis[start:stop]
            changed.coefficient_labels[start:stop] = gauge @ batch.coefficient_labels[start:stop]
            transformed[start:stop] = gauge @ coefficients[start:stop]
        original, _ = training_loss(Decoder(), coefficients, batch, "dielectric", space="tensor", kind="huber")
        actual, _ = training_loss(Decoder(), transformed, changed, "dielectric", space="tensor", kind="huber")
        torch.testing.assert_close(actual, original, atol=1e-6, rtol=1e-6)
        expected = torch.nn.functional.huber_loss(Decoder().reconstruct(coefficients, batch), batch.labels)
        torch.testing.assert_close(original, expected)
        first, _ = training_loss(Decoder(), coefficients, batch, "dielectric", kind="huber")
        second, _ = training_loss(Decoder(), transformed, changed, "dielectric", kind="huber")
        self.assertGreater(abs(float(first-second)), .01)

    def test_scalar_statistics_use_only_training_labels(self):
        data = [SimpleNamespace(labels=torch.eye(3)*v) for v in (2., 4., 6., 1e12)]
        stats = dielectric_scalar_statistics(data, [0, 1, 2])
        self.assertAlmostEqual(stats["shift"], 4*math.sqrt(3))
        self.assertAlmostEqual(stats["scale"], 2*math.sqrt(3))
        data[-1].labels[:] = -1e12
        self.assertEqual(stats, dielectric_scalar_statistics(data, [0, 1, 2]))

    def test_independent_scalar_head_respects_mixed_basis_gauges_and_o3(self):
        torch.manual_seed(17)
        irreps = o3.Irreps("3x0e+2x0o+2x1o+2x2e")
        batch = self.batch()
        batch.canonical_basis = batch.canonical_basis.double()
        batch.cartesian_basis = batch.cartesian_basis.double()
        head = SpaceGroupCoefficientHead("dielectric", irreps, hidden_dim=16,
                                        scalar_mode="independent", scalar_shift=3., scalar_scale=2.).double()
        # Exercise a learned, nonconstant scalar branch, not only its initializer.
        with torch.no_grad():
            head.scalar_readout[-1].weight.normal_(std=.2)
        features = torch.randn(2, irreps.dim, dtype=torch.float64)
        prediction = head(features, batch)
        changed = batch.clone()
        expected = []
        for start, stop in zip(batch.coefficient_ptr[:-1], batch.coefficient_ptr[1:]):
            gauge = torch.linalg.qr(torch.randn(int(stop-start), int(stop-start), dtype=torch.float64)).Q
            changed.canonical_basis[start:stop] = gauge @ batch.canonical_basis[start:stop]
            changed.cartesian_basis[start:stop] = gauge @ batch.cartesian_basis[start:stop]
            expected.append(gauge @ prediction[start:stop])
        actual = head(features, changed)
        torch.testing.assert_close(actual, torch.cat(expected), atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(Decoder().reconstruct(actual, changed), Decoder().reconstruct(prediction, batch), atol=1e-10, rtol=1e-10)
        for sign in (1, -1):
            rotation = o3.rand_matrix(dtype=torch.float64) * sign
            rotated = batch.clone()
            rotated.canonical_basis = torch.tensor(rotate_tensor(
                batch.canonical_basis.numpy().reshape(-1, 3, 3), rotation.numpy(), 2).reshape(-1, 9))
            torch.testing.assert_close(head(features @ irreps.D_from_matrix(rotation).T, rotated), prediction,
                                       atol=3e-6, rtol=3e-5)
        head(features, batch).square().sum().backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in head.parameters()))
        self.assertGreater(float(head.scalar_readout[-1].weight.grad.abs().sum()), 0)

    def test_scalar_head_checkpoint_round_trip_and_legacy_default(self):
        args = make_model_args("dielectric", self.dataset[0].num_node_features, small=True)
        legacy = IrredNetSpaceGroup(args)
        self.assertEqual(legacy.coefficient_head.scalar_mode, "contract")
        args.dielectric_scalar_head = "independent"
        args.dielectric_scalar_shift, args.dielectric_scalar_scale = 3., 2.
        model = IrredNetSpaceGroup(args).eval()
        checkpoint = {"model_args":vars(args), "model_state_dict":model.state_dict(), "version":MODEL_VERSION,
                      "basis_version":BASIS_VERSION, "label_convention":LABEL_CONVENTION}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"checkpoint.pt"
            torch.save(checkpoint,path)
            restored, _ = load_coefficient_model(path)
            restored.eval()
            with torch.no_grad():
                torch.testing.assert_close(restored(self.batch()), model(self.batch()), atol=0, rtol=0)

    def test_dielectric_error_components_are_reported_consistently(self):
        class ZeroPredictor(Decoder):
            def eval(self):
                return self
            def __call__(self, batch):
                return torch.zeros_like(batch.coefficient_labels)
        result = evaluate(ZeroPredictor(), [self.batch()], "cpu", "dielectric")
        self.assertAlmostEqual(result["dielectric_scalar"]["mae"], 3.)
        self.assertAlmostEqual(result["tensor"]["mse"], result["dielectric_scalar"]["mse"]/3
                               + result["dielectric_deviatoric"]["mse"], places=6)



if __name__ == "__main__":
    unittest.main()
