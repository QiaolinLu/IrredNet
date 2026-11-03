"""Deterministic, rotation-consistent coordinates for the allowed tensor space."""

from dataclasses import dataclass

import numpy as np
import spglib
import torch

from .decomposition import PROPERTY_FORMULAS, as_cell, tensor_basis, tensor_representation

BASIS_VERSION = "lattice_qr_o3_projector_mgs_v2"
MODEL_VERSION = "space_group_equivariant_coefficients_v3"
LABEL_CONVENTION = "jarvis_vasp_unscaled_xx_yy_zz_xy_yz_xz_v1"
PIEZO_PAIRS = ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2))
# elastic_total_kbar preserves VASP's XX YY ZZ XY YZ ZX columns.
# ZX and XZ coincide by minor symmetry. Do not use conventional Voigt order.
ELASTIC_PAIRS = ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2))


def tensor_shape(target):
    return (3,) * len(PROPERTY_FORMULAS[target].split("=")[0])


def rotate_tensor(tensor, rotation, rank):
    """Active orthogonal transformation on every Cartesian index; keeps batch axes."""
    left, right = "ijkl"[:rank], "mnop"[:rank]
    terms = ["..." + right] + [i + j for i, j in zip(left, right)]
    return np.einsum(
        ",".join(terms) + "->..." + left, tensor, *([rotation] * rank), optimize=True
    )


def lattice_frame(lattice):
    """L.T = F R with positive diagonal R; F may include a reflection.

    For a rotation/reflection O, F(L O.T) = O F(L), so L F is unchanged.
    Ordered lattice vectors define the gauge; this does not standardize a cell.
    """
    frame, upper = np.linalg.qr(np.asarray(lattice, dtype=float).T)
    signs = np.where(np.diag(upper) < 0, -1., 1.)
    return frame * signs[None, :], (signs[:, None] * upper).T


def _orthogonalize(vectors, expected, tol=1e-6):
    """Ordered modified Gram-Schmidt, with a stable first-nonzero sign rule."""
    accepted = []
    if expected == 0:
        return np.empty((0, vectors.shape[-1]), dtype=float)
    for vector in vectors:
        vector = vector.copy()
        for _ in range(2):
            for previous in accepted:
                vector -= np.dot(previous, vector) * previous
        norm = np.linalg.norm(vector)
        if norm < tol:
            continue
        vector /= norm
        pivot = np.flatnonzero(np.abs(vector) > tol)[0]
        if vector[pivot] < 0:
            vector *= -1
        accepted.append(vector)
        if len(accepted) == expected:
            break
    if len(accepted) != expected:
        raise RuntimeError("Could not construct the expected independent coefficient basis.")
    return np.stack(accepted)


@dataclass
class CoefficientBasis:
    target: str
    frame: np.ndarray
    canonical_lattice: np.ndarray
    canonical_basis: np.ndarray  # [r, 3**rank], orthonormal Cartesian tensors
    cartesian_basis: np.ndarray  # [r, 3**rank], in the original input frame
    symmetry: dict

    @property
    def dim(self):
        return len(self.cartesian_basis)

    def encode(self, tensor):
        tensor = np.asarray(tensor)
        shape = tensor_shape(self.target)
        if tensor.shape[-len(shape):] != shape:
            raise ValueError(f"Expected Cartesian tensor with trailing dimensions {shape}.")
        flat = tensor.reshape(tensor.shape[:-len(shape)] + (int(np.prod(shape)),))
        return flat @ self.cartesian_basis.T

    def decode(self, coefficients):
        coefficients = np.asarray(coefficients)
        if coefficients.ndim == 0 or coefficients.shape[-1] != self.dim:
            raise ValueError(f"Expected {self.dim} independent coefficients.")
        return (coefficients @ self.cartesian_basis).reshape(
            coefficients.shape[:-1] + tensor_shape(self.target)
        )


def build_coefficient_basis(structure, target, *, symprec=1e-5):
    """Extract only the trivial Gamma sector; full branching remains in report.py.

    Group averaging is the orthogonal invariant projector. Avoid enumerating
    every nontrivial irrep for every training sample, which is unnecessary for
    homogeneous tensor labels and was the main preprocessing bottleneck.
    """
    if not np.isfinite(symprec) or symprec <= 0:
        raise ValueError("symprec must be positive.")
    lattice, positions, numbers = as_cell(structure)
    frame, canonical_lattice = lattice_frame(lattice)
    irreps, cartesian = tensor_basis(target)
    dataset = spglib.get_symmetry_dataset((canonical_lattice, positions, numbers), symprec=symprec)
    if dataset is None:
        raise ValueError("spglib could not determine symmetry for this structure.")
    rotations = np.unique(dataset.rotations, axis=0)
    a = canonical_lattice.T
    cartesian_rotations = a @ rotations @ np.linalg.inv(a)
    if not np.allclose(cartesian_rotations @ cartesian_rotations.swapaxes(1, 2), np.eye(3), atol=1e-5, rtol=0):
        raise ValueError("Symmetry rotations are not orthogonal; check symprec/structure.")
    average = tensor_representation(cartesian, cartesian_rotations).mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh((average + average.T) / 2)
    allowed = eigenvalues > .5
    if np.max(np.abs(eigenvalues - allowed.astype(float))) > 1e-5:
        raise RuntimeError("Group average is not an invariant projector; check symmetry tolerance.")
    r = int(allowed.sum())
    projector = eigenvectors[:, allowed] @ eigenvectors[:, allowed].T
    # Projector columns fix the ordering inside repeated trivial irreps without
    # depending on eigenvector gauges or spgrep's irrep numbering.
    o3_basis = _orthogonalize(projector.T, r)
    shape = tensor_shape(target)
    cart_dim = int(np.prod(shape))
    q = cartesian.reshape(irreps.dim, cart_dim)
    # Remove small floating errors in e3nn's Cartesian basis as well.
    canonical_basis = _orthogonalize(o3_basis @ q, r)
    cartesian_basis = rotate_tensor(
        canonical_basis.reshape((r,) + shape), frame, len(shape)
    ).reshape(r, cart_dim)
    symmetry = {
        "target": target, "formula": PROPERTY_FORMULAS[target], "o3_irreps": str(irreps),
        "space_group_number": int(dataset.number), "international": dataset.international,
        "pointgroup": dataset.pointgroup, "kpoint": [0, 0, 0], "symprec": symprec,
        "tensor_dim": irreps.dim, "invariant_dim": r, "num_operations": len(dataset.rotations),
        "basis_version": BASIS_VERSION, "method": "Gamma group-average invariant projector",
    }
    return CoefficientBasis(target, frame, canonical_lattice, canonical_basis, cartesian_basis, symmetry)


def labels_to_cartesian(labels, target):
    """Expand the project's unscaled Voigt values; do not change the input."""
    labels = torch.as_tensor(labels)
    shape = tensor_shape(target)
    if labels.shape == shape:
        return labels.clone()
    if target == "piezoelectric" and labels.shape == (3, 6):
        result = labels.new_zeros(shape)
        for column, (j, k) in enumerate(PIEZO_PAIRS):
            result[:, j, k] = result[:, k, j] = labels[:, column]
        return result
    if target == "elastic" and labels.shape == (6, 6):
        mapping = torch.empty(3, 3, dtype=torch.long, device=labels.device)
        for column, (i, j) in enumerate(ELASTIC_PAIRS):
            mapping[i, j] = mapping[j, i] = column
        return labels[mapping[:, :, None, None], mapping[None, None, :, :]]
    raise ValueError(f"Unsupported {target} label shape {tuple(labels.shape)}.")


def reconstruct_tensors(coefficients, data, target):
    """Reconstruct tensors from packed, variable-length coefficient vectors."""
    if coefficients.ndim != 1 or len(coefficients) != len(data.coefficient_batch):
        raise ValueError("Expected one scalar per actual independent coefficient.")
    basis = data.cartesian_basis.to(coefficients)
    shape = tensor_shape(target)
    flat = coefficients.new_zeros(data.num_graphs, int(np.prod(shape)))
    flat = flat.index_add(0, data.coefficient_batch, coefficients[:, None] * basis)
    return flat.reshape((data.num_graphs,) + shape)


def coefficient_loss(prediction, labels, kind="mse"):
    """Every entry is an actual degree of freedom; no padding or fake zero labels."""
    if prediction.shape != labels.shape or prediction.ndim != 1:
        raise ValueError("Coefficient predictions and labels must be matching packed vectors.")
    if kind not in ("mse", "l1", "huber"):
        raise ValueError(f"Unknown coefficient loss {kind!r}.")
    if prediction.numel() == 0:
        return prediction.sum()
    functions = {"mse": torch.nn.functional.mse_loss, "l1": torch.nn.functional.l1_loss,
                 "huber": torch.nn.functional.huber_loss}
    return functions[kind](prediction, labels)
