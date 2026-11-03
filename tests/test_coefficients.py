import copy
import hashlib
import json
from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np
from pymatgen.core import Structure
import torch
from e3nn import o3
from torch.utils.data import DataLoader

from irrednet import PROPERTY_FORMULAS, decompose_property
from irrednet.coefficients import (
    BASIS_VERSION, MODEL_VERSION, LABEL_CONVENTION, build_coefficient_basis, coefficient_loss,
    labels_to_cartesian, reconstruct_tensors, rotate_tensor, tensor_shape,
)
from irrednet.data import SpaceGroupDataset, load_samples, match_label_shape
from irrednet.model import IrredNetSpaceGroup, SpaceGroupCoefficientHead, load_coefficient_model
from irrednet.predict import save_predictions
from irrednet.train import evaluate, make_model_args, split_indices
from .test_space_group import structures


class TestDirectCoefficients(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.cells = structures()

    def dataset(self, target, names=("Pm-3m", "P1"), labeled=True):
        shape = {"dielectric": (3, 3), "piezoelectric": (3, 6), "elastic": (6, 6)}[target]
        samples = [{"structure": self.cells[name][0], target: np.ones(shape), "id": name} for name in names]
        return SpaceGroupDataset(samples, target, require_labels=labeled)

    def test_known_counts_and_orthonormal_reconstruction_basis(self):
        for name, (structure, _, counts) in self.cells.items():
            for target, count in zip(PROPERTY_FORMULAS, counts):
                with self.subTest(name=name, target=target):
                    basis = build_coefficient_basis(structure, target)
                    self.assertEqual(basis.dim, count)
                    full = decompose_property(structure, target)
                    q = full.cartesian_basis.reshape(full.dim, -1)
                    np.testing.assert_allclose(basis.cartesian_basis.T @ basis.cartesian_basis,
                                               q.T @ full.projector @ q, atol=2e-6)
                    np.testing.assert_allclose(basis.cartesian_basis @ basis.cartesian_basis.T, np.eye(count), atol=1e-12)
                    coefficients = np.arange(count, dtype=float) + .25
                    np.testing.assert_allclose(basis.encode(basis.decode(coefficients)), coefficients, atol=1e-11)
        basis = build_coefficient_basis(self.cells["Pm-3m"][0], "dielectric")
        np.testing.assert_allclose(basis.encode(np.diag([5., 5., 8.])), [6 * np.sqrt(3)], atol=1e-6)
        np.testing.assert_allclose(basis.decode(basis.encode(np.diag([5., 5., 8.]))), np.eye(3) * 6, atol=1e-6)

    def test_packed_counts_have_no_padded_targets(self):
        for target, counts in (("dielectric", [1, 6]), ("piezoelectric", [0, 18]), ("elastic", [3, 21])):
            dataset = self.dataset(target)
            batch = dataset.collate([dataset[0], dataset[1]])
            self.assertEqual(batch.num_coefficients.tolist(), counts)
            self.assertEqual(batch.coefficient_ptr.tolist(), [0, counts[0], sum(counts)])
            self.assertEqual(batch.coefficient_labels.shape, (sum(counts),))
            self.assertEqual(batch.cartesian_basis.shape, (sum(counts), int(np.prod(tensor_shape(target)))))
            self.assertEqual(batch.space_group_number.tolist(), [221, 1])

    def test_voigt_expansion_preserves_project_orders(self):
        piezo = torch.arange(18.).reshape(3, 6)
        full = labels_to_cartesian(piezo, "piezoelectric")
        self.assertEqual(full[0, 0, 1], piezo[0, 3])
        self.assertEqual(full[0, 1, 2], piezo[0, 4])
        self.assertEqual(full[0, 0, 2], piezo[0, 5])
        torch.testing.assert_close(full, full.transpose(1, 2))
        torch.testing.assert_close(match_label_shape(full[None], piezo[None], "piezoelectric"), piezo[None])
        elastic = torch.arange(36.).reshape(6, 6)
        full = labels_to_cartesian(elastic, "elastic")
        self.assertEqual(full[0, 0, 0, 1], elastic[0, 3])
        self.assertEqual(full[0, 0, 1, 2], elastic[0, 4])
        self.assertEqual(full[0, 0, 0, 2], elastic[0, 5])
        torch.testing.assert_close(match_label_shape(full[None], elastic[None], "elastic"), elastic[None])

    def test_synthetic_hexagonal_vasp_label_is_expressible(self):
        # Analytic hexagonal tensor; no measured material or external fixture.
        structure = self.cells["P6/mmm"][0]
        labels = np.array([[180., 60., 45., 0., 0., 0.],
                           [60., 180., 45., 0., 0., 0.],
                           [45., 45., 220., 0., 0., 0.],
                           [0., 0., 0., 60., 0., 0.],
                           [0., 0., 0., 0., 70., 0.],
                           [0., 0., 0., 0., 0., 70.]])
        self.assertEqual(labels[3, 3], (labels[0, 0] - labels[0, 1]) / 2)
        tensor = labels_to_cartesian(labels, "elastic").numpy()
        self.assertEqual(tensor[0, 1, 0, 1], 60.)
        self.assertEqual(tensor[1, 2, 1, 2], 70.)
        basis = build_coefficient_basis(structure, "elastic")
        self.assertEqual(basis.dim, 5)
        np.testing.assert_allclose(basis.decode(basis.encode(tensor)), tensor, atol=1e-4)
        dataset = SpaceGroupDataset([{"structure": structure, "elastic": labels,
                                     "id": "synthetic-hexagonal"}], "elastic")
        self.assertEqual(dataset[0].sample_id, "synthetic-hexagonal")

    def test_label_residual_and_coefficient_frobenius_metric(self):
        dataset = self.dataset("dielectric", ("Pm-3m",))
        batch = dataset.collate([dataset[0]])
        projected = reconstruct_tensors(batch.coefficient_labels, batch, "dielectric")
        # All-ones dielectric labels are not allowed by cubic symmetry; off-diagonal
        # violations remain observable and must not become fake coefficient targets.
        torch.testing.assert_close(projected, torch.eye(3)[None], atol=1e-6, rtol=0)
        torch.testing.assert_close(batch.label_projection_residual, torch.tensor([np.sqrt(6)], dtype=torch.float32))
        for target in PROPERTY_FORMULAS:
            dataset = self.dataset(target)
            batch = dataset.collate([dataset[0], dataset[1]])
            delta = torch.linspace(.1, 1., batch.coefficient_labels.numel())
            tensor_delta = reconstruct_tensors(delta, batch, target)
            torch.testing.assert_close(tensor_delta.square().sum(), delta.square().sum(), atol=1e-5, rtol=1e-6)

    def test_source_files_labels_and_stale_metadata_untouched(self):
        samples = [{"structure": self.cells["Pm-3m"][0], "dielectric": np.arange(9).reshape(3, 3),
                    "invariant_dim": 999, "invariant_basis": np.ones((6, 6))}]
        before = copy.deepcopy(samples)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "preprocessed.pkl"
            path.write_bytes(pickle.dumps(samples))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            loaded = load_samples(path)
            dataset = SpaceGroupDataset(loaded, "dielectric")
            self.assertEqual(dataset[0].num_coefficients, 1)
            self.assertEqual(loaded[0]["invariant_dim"], 999)
            self.assertEqual(set(loaded[0]), set(before[0]))
            np.testing.assert_array_equal(loaded[0]["dielectric"], before[0]["dielectric"])
            np.testing.assert_array_equal(loaded[0]["invariant_basis"], before[0]["invariant_basis"])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)

    def test_each_model_directly_predicts_coefficients_and_trains(self):
        for target in PROPERTY_FORMULAS:
            with self.subTest(target=target):
                dataset = self.dataset(target)
                batch = dataset.collate([dataset[0], dataset[1]])
                model = IrredNetSpaceGroup(make_model_args(target, batch.num_node_features, small=True))
                observed_shapes = []
                hooks = [path.register_forward_hook(
                    lambda module, args, output: observed_shapes.append(tuple(output.shape)))
                    for path in model.coefficient_head.paths]
                prediction = model(batch)
                for hook in hooks:
                    hook.remove()
                self.assertEqual(prediction.shape, batch.coefficient_labels.shape)
                self.assertTrue(observed_shapes)
                self.assertTrue(all(shape == tuple(prediction.shape) for shape in observed_shapes))
                self.assertFalse(hasattr(model.TI, "extra_layers"))
                loss = coefficient_loss(prediction, batch.coefficient_labels)
                loss.backward()
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                self.assertTrue(all(torch.isfinite(grad).all() for grad in gradients))
                self.assertGreater(sum(grad.abs().sum().item() for grad in gradients), 0)
                torch.optim.Adam(model.parameters(), lr=1e-3).step()
                tensor = model.reconstruct(prediction, batch)
                self.assertEqual(tuple(tensor.shape), (2,) + tensor_shape(target))
                if target == "piezoelectric":
                    torch.testing.assert_close(tensor[0], torch.zeros_like(tensor[0]), rtol=0, atol=0)
                batch.sg_target = ["incorrect"]
                with self.assertRaisesRegex(ValueError, "metadata"):
                    model(batch)

    def test_zero_coefficient_batch_has_finite_loss_and_no_fake_predictions(self):
        dataset = self.dataset("piezoelectric", ("Pm-3m", "P-1"))
        batch = dataset.collate([dataset[0], dataset[1]])
        model = IrredNetSpaceGroup(make_model_args("piezoelectric", batch.num_node_features, small=True))
        prediction = model(batch)
        self.assertEqual(prediction.numel(), 0)
        self.assertEqual(batch.coefficient_ptr.tolist(), [0, 0, 0])
        loss = coefficient_loss(prediction, batch.coefficient_labels)
        self.assertEqual(loss.item(), 0)
        loss.backward()
        torch.testing.assert_close(model.reconstruct(prediction, batch), torch.zeros(2, 3, 3, 3), atol=0, rtol=0)
        metrics = evaluate(model, DataLoader(dataset, batch_size=2, collate_fn=dataset.collate), "cpu", "piezoelectric")
        self.assertEqual(metrics["coefficient"]["count"], 0)
        self.assertEqual(metrics["coefficient"]["num_materials"], 0)
        self.assertIsNone(metrics["coefficient"]["ewt10"])
        self.assertEqual(metrics["tensor"]["mae"], 1)
        self.assertEqual(metrics["tensor"]["ewt25"], 0)
        self.assertEqual(metrics["tensor_zero_dof"]["num_materials"], 2)
        self.assertEqual(metrics["tensor_zero_dof"]["ewt25"], 0)
        self.assertEqual(metrics["tensor_nonzero_dof"]["num_materials"], 0)
        self.assertIsNone(metrics["tensor_nonzero_dof"]["ewt25"])

    def test_piezo_metrics_separate_forbidden_tensors_from_learnable_tensors(self):
        dataset = self.dataset("piezoelectric", ("Pm-3m", "P1"))
        dataset.graphs[0].labels.zero_()
        dataset.graphs[0].cartesian_labels.zero_()

        class ZeroPrediction(torch.nn.Module):
            def forward(self, batch):
                return torch.zeros_like(batch.coefficient_labels)

            def reconstruct(self, coefficients, batch):
                return reconstruct_tensors(coefficients, batch, "piezoelectric")

        metrics = evaluate(ZeroPrediction(), DataLoader(dataset, batch_size=2,
                           collate_fn=dataset.collate), "cpu", "piezoelectric")
        self.assertEqual(metrics["tensor"]["ewt10"], .5)
        self.assertEqual(metrics["tensor_zero_dof"]["ewt10"], 1.)
        self.assertEqual(metrics["tensor_nonzero_dof"]["ewt10"], 0.)
        self.assertEqual(metrics["tensor_nonzero_dof"]["num_materials"], 1)

    def test_ewt_uses_per_material_vectors_not_per_component_errors(self):
        class KnownPrediction(torch.nn.Module):
            def forward(self, batch):
                # Three relative vector errors: 0%, 8%, 20%.
                factor = torch.tensor([1., 1.08, 1.20])[batch.coefficient_batch]
                return batch.coefficient_labels * factor

            def reconstruct(self, coefficients, batch):
                return reconstruct_tensors(coefficients, batch, "dielectric")

        dataset = SpaceGroupDataset([
            {"structure": self.cells["Pm-3m"][0], "dielectric": np.eye(3) * 10},
            {"structure": self.cells["P6/mmm"][0], "dielectric": np.eye(3) * 10},
            {"structure": self.cells["P1"][0], "dielectric": np.eye(3) * 10},
        ], "dielectric")
        metrics = evaluate(KnownPrediction(), DataLoader(dataset, batch_size=3,
                           collate_fn=dataset.collate), "cpu", "dielectric")
        for name in ("coefficient", "tensor"):
            self.assertEqual(metrics[name]["num_materials"], 3)
            self.assertEqual(metrics[name]["ewt25"], 1)
            self.assertAlmostEqual(metrics[name]["ewt10"], 2 / 3)
            self.assertAlmostEqual(metrics[name]["ewt05"], 1 / 3)

    def test_isolated_atoms_train_alone_and_in_mixed_batches(self):
        isolated = Structure(np.eye(3) * 20, ["Na"], [[.5, .5, .5]])
        samples = [{"structure": isolated, "elastic": np.eye(6), "id": "isolated-Na"},
                   {"structure": self.cells["Pm-3m"][0], "elastic": np.eye(6), "id": "crystal"}]
        dataset = SpaceGroupDataset(samples, "elastic")
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset[0].num_edges, 0)
        self.assertGreater(dataset[1].num_edges, 0)
        self.assertEqual(dataset[0].num_coefficients, 3)
        alone = dataset.collate([dataset[0]])
        self.assertEqual(tuple(alone.edge_vec.shape), (0, 3))
        mixed = dataset.collate([dataset[1], dataset[0]])
        for interaction_type in ("standard",):
            with self.subTest(interaction_type=interaction_type):
                model = IrredNetSpaceGroup(make_model_args(
                    "elastic", alone.num_node_features, small=True, interaction_type=interaction_type))
                for batch in (alone, mixed):
                    model.zero_grad(set_to_none=True)
                    prediction = model(batch)
                    self.assertEqual(prediction.shape, batch.coefficient_labels.shape)
                    self.assertTrue(torch.isfinite(prediction).all())
                    loss = coefficient_loss(prediction, batch.coefficient_labels)
                    loss.backward()
                    gradients = [p.grad for p in model.parameters() if p.grad is not None]
                    self.assertTrue(all(torch.isfinite(grad).all() for grad in gradients))
                    self.assertGreater(sum(grad.abs().sum().item() for grad in gradients), 0)
                with torch.no_grad():
                    torch.testing.assert_close(model(alone), model(mixed)[-3:], atol=1e-5, rtol=1e-5)

    def test_rotations_and_reflections_preserve_coefficients_and_rotate_tensors(self):
        angle = .413
        rz = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1.]])
        rx = np.array([[1., 0, 0], [0, np.cos(.271), -np.sin(.271)], [0, np.sin(.271), np.cos(.271)]])
        for target in PROPERTY_FORMULAS:
            structure = self.cells["P3"][0]  # skew lattice and multiple invariant sectors
            basis = build_coefficient_basis(structure, target)
            label = basis.decode(np.linspace(.3, 1.1, basis.dim))
            first = SpaceGroupDataset([{"structure": structure, target: label}], target)
            first_batch = first.collate([first[0]])
            model = IrredNetSpaceGroup(make_model_args(target, first_batch.num_node_features, small=True)).eval()
            with torch.no_grad():
                first_coefficients = model(first_batch)
                first_tensor = model.reconstruct(first_coefficients, first_batch).numpy()[0]
            for rotation in (rz @ rx, np.diag([-1., 1, 1]) @ rz @ rx):
                with self.subTest(target=target, determinant=np.linalg.det(rotation)):
                    rotated = Structure(structure.lattice.matrix @ rotation.T, structure.species, structure.frac_coords)
                    rotated_label = rotate_tensor(label, rotation, len(tensor_shape(target)))
                    second = SpaceGroupDataset([{"structure": rotated, target: rotated_label}], target)
                    batch = second.collate([second[0]])
                    torch.testing.assert_close(batch.canonical_basis, first_batch.canonical_basis, atol=2e-6, rtol=0)
                    torch.testing.assert_close(batch.coefficient_labels, first_batch.coefficient_labels, atol=2e-6, rtol=0)
                    with torch.no_grad():
                        coefficients = model(batch)
                        tensor = model.reconstruct(coefficients, batch).numpy()[0]
                    torch.testing.assert_close(coefficients, first_coefficients, atol=5e-5, rtol=5e-5)
                    np.testing.assert_allclose(tensor, rotate_tensor(first_tensor, rotation, len(tensor_shape(target))), atol=5e-5)

    def test_head_is_covariant_under_basis_gauge_and_invariant_under_o3(self):
        torch.manual_seed(17)
        irreps = o3.Irreps("3x0e+2x0o+2x1o+2x2e+2x2o+1x3o+1x4e")
        for target in PROPERTY_FORMULAS:
            with self.subTest(target=target):
                dataset = self.dataset(target, ("P3", "P1"), labeled=False)
                batch = dataset.collate([dataset[0], dataset[1]])
                batch.canonical_basis = batch.canonical_basis.double()
                batch.cartesian_basis = batch.cartesian_basis.double()
                head = SpaceGroupCoefficientHead(target, irreps, hidden_dim=16).double()
                features = torch.randn(2, irreps.dim, dtype=torch.float64)
                prediction = head(features, batch)
                tensor = reconstruct_tensors(prediction, batch, target)
                changed = batch.clone()
                expected = []
                for start, stop in zip(batch.coefficient_ptr[:-1], batch.coefficient_ptr[1:]):
                    size = int(stop - start)
                    gauge = torch.linalg.qr(torch.randn(size, size, dtype=torch.float64)).Q
                    changed.canonical_basis[start:stop] = gauge @ batch.canonical_basis[start:stop]
                    changed.cartesian_basis[start:stop] = gauge @ batch.cartesian_basis[start:stop]
                    expected.append(gauge @ prediction[start:stop])
                actual = head(features, changed)
                torch.testing.assert_close(actual, torch.cat(expected), atol=1e-12, rtol=1e-11)
                torch.testing.assert_close(reconstruct_tensors(actual, changed, target), tensor, atol=1e-12, rtol=1e-11)
                for sign in (1, -1):
                    rotation = o3.rand_matrix(dtype=torch.float64) * sign
                    rotated = batch.clone()
                    rotated.canonical_basis = torch.tensor(rotate_tensor(
                        batch.canonical_basis.numpy().reshape((-1,) + tensor_shape(target)),
                        rotation.numpy(), len(tensor_shape(target))).reshape(batch.canonical_basis.shape))
                    rotated_features = features @ irreps.D_from_matrix(rotation).T
                    torch.testing.assert_close(head(rotated_features, rotated), prediction, atol=2e-6, rtol=2e-5)

    def test_equivalent_cells_preserve_predicted_physical_tensor(self):
        torch.manual_seed(31)
        transforms = (np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]),
                      np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]]),
                      np.diag([-1, 1, 1]))
        for target in PROPERTY_FORMULAS:
            model = None
            for name in ("P1", "P3", "P6/mmm", "F-43m"):
                original = self.cells[name][0]
                cells = [original] + [Structure(
                    u @ original.lattice.matrix, original.species, original.frac_coords @ np.linalg.inv(u),
                    to_unit_cell=False) for u in transforms]
                if name == "F-43m":
                    cells.append(original.get_primitive_structure())
                dataset = SpaceGroupDataset([{"structure": s} for s in cells], target, require_labels=False)
                batch = dataset.collate([dataset[i] for i in range(len(cells))])
                if model is None:
                    model = IrredNetSpaceGroup(make_model_args(target, batch.num_node_features, small=True)).eval()
                with self.subTest(target=target, crystal=name), torch.no_grad():
                    tensors = model.predict_tensor(batch)
                    for other in tensors[1:]:
                        torch.testing.assert_close(other, tensors[0], atol=2e-5, rtol=2e-4)

    def test_checkpoint_and_unlabeled_prediction_export_roundtrip(self):
        dataset = self.dataset("elastic", labeled=False)
        self.assertNotIn("labels", dataset[0])
        args = make_model_args("elastic", dataset[0].num_node_features, small=True)
        model = IrredNetSpaceGroup(args)
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "best.pt"
            torch.save({"model_state_dict": model.state_dict(), "model_args": vars(args),
                        "version": MODEL_VERSION, "basis_version": BASIS_VERSION,
                        "label_convention": LABEL_CONVENTION, "symprec": 1e-5}, checkpoint)
            loaded, _ = load_coefficient_model(checkpoint)
            loader = DataLoader(dataset, batch_size=1, collate_fn=dataset.collate)
            output = Path(folder) / "predictions.npz"
            save_predictions(loaded, loader, "cpu", output)
            with np.load(output, allow_pickle=False) as result:
                self.assertNotIn("coefficient_labels", result.files)
                self.assertEqual(str(result["label_convention"]), LABEL_CONVENTION)
                self.assertEqual(str(result["model_version"]), MODEL_VERSION)
                self.assertEqual(result["coefficient_ptr"].tolist(), [0, 3, 24])
                for index in range(2):
                    start, stop = result["coefficient_ptr"][index:index+2]
                    reconstructed = result["coefficients"][start:stop] @ result["cartesian_basis"][start:stop]
                    np.testing.assert_allclose(reconstructed.reshape(3, 3, 3, 3), result["cartesian_tensor"][index], atol=1e-6)
            bad = Path(folder) / "old.pt"
            torch.save({"version": "space_group_v1"}, bad)
            with self.assertRaisesRegex(ValueError, "retrained"):
                load_coefficient_model(bad)
            stale = torch.load(checkpoint, weights_only=True)
            stale.pop("label_convention")
            torch.save(stale, bad)
            with self.assertRaisesRegex(ValueError, "retrained"):
                load_coefficient_model(bad)

    def test_split_keeps_learning_samples_and_no_overlap(self):
        counts = [0] * 17 + [1, 2, 3]
        splits = split_indices(counts, 32)
        self.assertEqual(sorted(np.concatenate(list(splits.values())).tolist()), list(range(20)))
        for ids in splits.values():
            self.assertTrue(any(counts[i] for i in ids))
        splits = split_indices([0, 0, 0, 1], 32)
        self.assertIn(3, splits["train"])


if __name__ == "__main__":
    unittest.main()
