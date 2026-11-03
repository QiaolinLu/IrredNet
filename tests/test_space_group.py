import unittest
from types import SimpleNamespace

import numpy as np
from pymatgen.core import Lattice, Structure

from irrednet import PROPERTY_FORMULAS, decompose_property
from irrednet.decomposition import as_cell, tensor_basis, tensor_representation


def structures():
    triclinic = Lattice([[3.1, 0, 0], [.2, 4.2, 0], [.4, .3, 5.3]])
    general = [[.13, .23, .17], [.31, .16, .63]]
    return {
        "P1": (Structure(triclinic, [6, 8, 14], [[.12, .23, .34], [.35, .19, .81], [.51, .41, .24]]), 1, (6, 18, 21)),
        "P-1": (Structure(triclinic, [14], [[0, 0, 0]]), 2, (6, 0, 21)),
        "Pm-3m": (Structure(Lattice.cubic(4), [14], [[0, 0, 0]]), 221, (1, 0, 3)),
        "P6/mmm": (Structure(Lattice.hexagonal(3.2, 5.1), [14], [[0, 0, 0]]), 191, (2, 0, 5)),
        "P4mm": (Structure.from_spacegroup("P4mm", Lattice.tetragonal(3.2, 5.1), [14, 8], general), 99, (2, 3, 6)),
        "P41": (Structure.from_spacegroup("P4_1", Lattice.tetragonal(3.2, 5.1), [14, 8], general), 76, (2, 4, 7)),
        "P3": (Structure.from_spacegroup("P3", Lattice.hexagonal(3.2, 5.1), [14, 8], general), 143, (2, 6, 7)),
        "F-43m": (Structure.from_spacegroup("F-43m", Lattice.cubic(5.4), [30, 16], [[0, 0, 0], [.25, .25, .25]]), 216, (1, 1, 3)),
        "rutile": (Structure.from_spacegroup("P4_2/mnm", Lattice.tetragonal(4.603, 2.969), [22, 8], [[0, 0, 0], [.3046, .3046, 0]]), 136, (2, 0, 6)),
    }


class TestSpaceGroupDecomposition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cells = structures()
        cls.results = {(name, target): decompose_property(structure, target)
                       for name, (structure, _, _) in cls.cells.items()
                       for target in PROPERTY_FORMULAS}

    def test_representation_matches_explicit_cartesian_index_action(self):
        rng = np.random.default_rng(7)
        rotation = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        rotations = np.stack((rotation, -rotation))
        for target in PROPERTY_FORMULAS:
            _, q = tensor_basis(target)
            rank = q.ndim - 1
            left, right = "ijkl"[:rank], "mnop"[:rank]
            terms = ["a" + left] + ["g" + i + j for i, j in zip(left, right)] + ["b" + right]
            reference = np.einsum(
                ",".join(terms) + "->gab", q, *([rotations] * rank), q, optimize=True)
            np.testing.assert_allclose(tensor_representation(q, rotations), reference, atol=1e-12)

    def test_legacy_sites_preserve_cell_and_reject_partial_occupancy(self):
        from pymatgen.core import Element

        class LegacyStructure:
            def __init__(self):
                self.lattice = SimpleNamespace(matrix=np.array([[3., 0, 0], [.2, 4., 0], [.3, .1, 5.]]))
                self.frac_coords = np.array([[.1, .2, .3], [.4, .5, .6]])
                self.sites = [SimpleNamespace(_species=SimpleNamespace(_data={Element("Si"): 1.})),
                              SimpleNamespace(_species=SimpleNamespace(_data={Element("O"): 1.}))]

            def __iter__(self):
                return iter(self.sites)

        legacy = LegacyStructure()
        lattice, positions, numbers = as_cell(legacy)
        np.testing.assert_array_equal(lattice, legacy.lattice.matrix)
        np.testing.assert_array_equal(positions, legacy.frac_coords)
        np.testing.assert_array_equal(numbers, [14, 8])
        self.assertFalse(hasattr(legacy, "is_ordered"))
        legacy.sites[0]._species._data[Element("Si")] = .5
        with self.assertRaisesRegex(ValueError, "full occupancy"):
            as_cell(legacy)
        legacy.sites[0]._species._data[Element("O")] = .5
        with self.assertRaisesRegex(ValueError, "one full-occupancy"):
            as_cell(legacy)

    def test_known_space_groups_and_independent_components(self):
        for name, (_, number, dimensions) in self.cells.items():
            for target, expected in zip(PROPERTY_FORMULAS, dimensions):
                with self.subTest(crystal=name, target=target):
                    result = self.results[name, target]
                    self.assertEqual(result.number, number)
                    self.assertEqual(result.invariant_dim, expected)

    def test_all_sectors_are_complete_and_transform_as_their_irreps(self):
        for key, result in self.results.items():
            with self.subTest(key=key):
                np.testing.assert_allclose(result.basis.conj().T @ result.basis, np.eye(result.dim), atol=2e-6)
                self.assertEqual(sum(s.irrep_dim * s.multiplicity for s in result.sectors), result.dim)
                for sector in result.sectors:
                    for start in range(sector.start, sector.stop, sector.irrep_dim):
                        u = result.basis[:, start:start + sector.irrep_dim]
                        np.testing.assert_allclose(result.representation @ u,
                                                   u @ result.irreps[sector.sg_irrep], atol=2e-6)

    def test_complex_irreps_roundtrip_and_real_reconstruction(self):
        rng = np.random.default_rng(19)
        result = self.results["P3", "elastic"]
        self.assertTrue(any(np.max(np.abs(ir.imag)) > .1 for ir in result.irreps))
        for result in self.results.values():
            coefficients = rng.normal(size=(4, result.dim))
            reconstructed = result.reconstruct(result.decompose(coefficients))
            np.testing.assert_allclose(reconstructed, coefficients, atol=2e-6)
            cartesian = result.to_cartesian(coefficients)
            np.testing.assert_allclose(result.from_cartesian(cartesian), coefficients, atol=2e-6)

    def test_projectors_idempotent_and_group_invariant(self):
        for key, result in self.results.items():
            with self.subTest(key=key):
                p = result.projector
                np.testing.assert_allclose(p @ p, p, atol=2e-6)
                np.testing.assert_allclose(p, result.representation.mean(axis=0), atol=2e-6)
                np.testing.assert_allclose(result.representation @ p,
                                           np.broadcast_to(p, result.representation.shape), atol=2e-6)

    def test_cubic_dielectric_is_trace_over_three_and_centrosymmetric_piezo_zero(self):
        tensor = np.arange(9).reshape(3, 3)
        result = self.results["Pm-3m", "dielectric"]
        np.testing.assert_allclose(result.project_tensor(tensor), np.eye(3) * np.trace(tensor) / 3, atol=2e-6)
        result = self.results["P-1", "piezoelectric"]
        np.testing.assert_array_equal(result.project_tensor(np.ones((3, 3, 3))), 0)

    def test_nonsymmorphic_translations_retained_but_trivial_at_gamma(self):
        for name in ("P41", "rutile"):
            result = self.results[name, "elastic"]
            self.assertTrue(np.any(np.abs(result.translations - np.round(result.translations)) > .1))
            np.testing.assert_allclose(result.projector, result.representation.mean(axis=0), atol=2e-6)

    def test_centered_conventional_cell_matches_primitive_cell(self):
        conventional = self.cells["F-43m"][0]
        primitive = conventional.get_primitive_structure()
        self.assertLess(len(primitive), len(conventional))
        for target in PROPERTY_FORMULAS:
            first = self.results["F-43m", target]
            second = decompose_property(primitive, target)
            self.assertEqual(first.number, second.number)
            # get_primitive_structure preserves the Cartesian orientation.
            np.testing.assert_allclose(first.projector, second.projector, atol=2e-6)

    def test_projector_covariance_in_rotated_and_reflected_skew_frames(self):
        angle = .413
        q = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        r = np.array([[1, 0, 0], [0, np.cos(.271), -np.sin(.271)], [0, np.sin(.271), np.cos(.271)]])
        for orientation in (q @ r, np.diag([-1., 1, 1]) @ q @ r):
            for target in PROPERTY_FORMULAS:
                structure = self.cells["P3"][0]
                rotated = Structure(structure.lattice.matrix @ orientation.T,
                                    structure.species, structure.frac_coords)
                original = self.results["P3", target]
                result = decompose_property(rotated, target)
                action = tensor_representation(original.cartesian_basis, orientation[None])[0]
                np.testing.assert_allclose(result.projector, action @ original.projector @ action.T, atol=3e-6)

    def test_non_gamma_invalid_cells_and_unknown_properties_rejected(self):
        structure = self.cells["Pm-3m"][0]
        with self.assertRaisesRegex(NotImplementedError, "Gamma"):
            decompose_property(structure, "dielectric", kpoint=[.5, 0, 0])
        with self.assertRaises(ValueError):
            decompose_property(structure, "dielectric", kpoint=[np.nan, 0, 0])
        with self.assertRaises(ValueError):
            decompose_property(structure, "magnetic")
        with self.assertRaises(ValueError):
            decompose_property((np.zeros((3, 3)), [[0, 0, 0]], [14]), "dielectric")


if __name__ == "__main__":
    unittest.main()
