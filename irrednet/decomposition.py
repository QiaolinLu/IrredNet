"""O(3) -> space-group irreps at Gamma for homogeneous polar tensors.

Fractional translations are retained in the symmetry metadata. At k=0 they act
trivially on these bulk tensors, including for nonsymmorphic space groups.
Irrep indices are local spgrep indices, not Bilbao/Mulliken labels.
"""

from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np
import spglib
import torch
from e3nn.io import CartesianTensor

PROPERTY_FORMULAS = {
    "dielectric": "ij=ji",
    "piezoelectric": "ijk=ikj",
    "elastic": "ijkl=ijlk=jikl=klij",
}


@lru_cache(maxsize=3)
def tensor_basis(target):
    """Return the existing e3nn coefficient convention and its Cartesian basis."""
    if target not in PROPERTY_FORMULAS:
        raise ValueError(f"Unknown property {target!r}; choose {tuple(PROPERTY_FORMULAS)}.")
    formula = PROPERTY_FORMULAS[target]
    ct = CartesianTensor(formula)
    rank = len(formula.split("=")[0])
    dummy = torch.empty((3,) * rank, dtype=torch.float64)
    q = ct.reduced_tensor_products(dummy).change_of_basis.detach().cpu().numpy()
    return ct, q


def as_cell(structure):
    """Accept a pymatgen Structure or (row lattice, fractional positions, species)."""
    if hasattr(structure, "lattice"):
        try:
            if not structure.is_ordered:
                raise ValueError("Disordered structures require an explicit ordered model.")
            numbers = structure.atomic_numbers
        except AttributeError:
            # Legacy dielectric pickles can contain old pymatgen sites
            # whose Composition objects lack current fields. Read their stored
            # species/occupancies without mutating or standardizing the crystal.
            from pymatgen.core import Element
            numbers = []
            for site in structure:
                composition = getattr(site, "__dict__", {}).get("_species")
                species = getattr(composition, "__dict__", {}).get("_data")
                if not isinstance(species, dict) or len(species) != 1:
                    raise ValueError("Legacy sites require one full-occupancy species; disordered sites are unsupported.")
                element, occupancy = next(iter(species.items()))
                if not np.isclose(occupancy, 1., atol=1e-8, rtol=0):
                    raise ValueError("Legacy sites require full occupancy; disordered sites are unsupported.")
                numbers.append(Element(getattr(element, "symbol", str(element))).Z)
        cell = (structure.lattice.matrix, structure.frac_coords, numbers)
    else:
        cell = structure
    if len(cell) != 3:
        raise ValueError("Expected (lattice, fractional positions, atomic numbers).")
    lattice, positions, numbers = (np.asarray(x) for x in cell)
    if lattice.shape != (3, 3) or not np.isfinite(lattice).all():
        raise ValueError("Lattice must be a finite 3x3 row-vector matrix.")
    if abs(np.linalg.det(lattice)) < 1e-12:
        raise ValueError("Lattice is singular.")
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) == 0:
        raise ValueError("Fractional positions must have shape [num_atoms, 3].")
    if numbers.shape != (len(positions),) or not np.isfinite(positions).all():
        raise ValueError("Atomic numbers and finite fractional positions must match.")
    if not np.isfinite(numbers).all() or not np.equal(numbers, numbers.astype(int)).all():
        raise ValueError("Atomic species identifiers must be integers.")
    return lattice.astype(float), positions.astype(float), numbers.astype(int)


def tensor_representation(q, rotations):
    """Restrict Cartesian tensor action to the e3nn basis (includes reflections)."""
    rank = q.ndim - 1
    # Transform one tensor index at a time. A single high-order einsum can
    # select a contraction with costly group-dependent outer products (rank 4
    # dominated the old preprocessing time). Intermediates here stay G*d*3^rank.
    transformed = np.broadcast_to(q, (len(rotations),) + q.shape)
    for axis in range(2, rank + 2):
        moved = np.moveaxis(transformed, axis, -1)
        moved = np.einsum("g...j,gij->g...i", moved, rotations, optimize=True)
        transformed = np.moveaxis(moved, -1, axis)
    flat = transformed.reshape(len(rotations), len(q), -1)
    return (flat @ q.reshape(len(q), -1).T).swapaxes(-1, -2)


@dataclass(frozen=True)
class IrrepSector:
    o3_irrep: str
    o3_copy: int
    sg_irrep: int
    irrep_dim: int
    multiplicity: int
    start: int
    stop: int
    trivial: bool


@dataclass
class SpaceGroupDecomposition:
    target: str
    formula: str
    o3_irreps: str
    number: int
    international: str
    pointgroup: str
    symprec: float
    rotations: np.ndarray
    translations: np.ndarray
    cartesian_rotations: np.ndarray
    equivalent_atoms: np.ndarray
    irreps: tuple
    representation: np.ndarray
    basis: np.ndarray
    sectors: tuple
    cartesian_basis: np.ndarray

    @property
    def dim(self):
        return self.basis.shape[0]

    @property
    def invariant_mask(self):
        mask = np.zeros(self.dim, dtype=bool)
        for sector in self.sectors:
            if sector.trivial:
                mask[sector.start:sector.stop] = True
        return mask

    @property
    def invariant_basis(self):
        # Trivial copies are explicitly made real during construction.
        return self.basis[:, self.invariant_mask].real

    @property
    def invariant_dim(self):
        return int(self.invariant_mask.sum())

    @property
    def projector(self):
        b = self.invariant_basis
        return b @ b.T

    def decompose(self, coefficients):
        """[..., O(3) dim] -> [..., SG dim], retaining complex conjugate irreps."""
        coefficients = np.asarray(coefficients)
        if coefficients.ndim < 1 or coefficients.shape[-1] != self.dim:
            raise ValueError(f"Expected coefficients with last dimension {self.dim}.")
        return coefficients @ self.basis.conj()

    def reconstruct(self, coefficients):
        coefficients = np.asarray(coefficients)
        if coefficients.ndim < 1 or coefficients.shape[-1] != self.dim:
            raise ValueError(f"Expected coefficients with last dimension {self.dim}.")
        return coefficients @ self.basis.T

    def from_cartesian(self, tensor):
        tensor = np.asarray(tensor)
        shape = self.cartesian_basis.shape[1:]
        if tensor.shape[-len(shape):] != shape:
            raise ValueError(f"Expected trailing Cartesian dimensions {shape}.")
        flat = tensor.reshape(tensor.shape[:-len(shape)] + (-1,))
        return flat @ self.cartesian_basis.reshape(self.dim, -1).T

    def to_cartesian(self, coefficients):
        coefficients = np.asarray(coefficients)
        if coefficients.ndim < 1 or coefficients.shape[-1] != self.dim:
            raise ValueError(f"Expected coefficients with last dimension {self.dim}.")
        flat = coefficients @ self.cartesian_basis.reshape(self.dim, -1)
        return flat.reshape(coefficients.shape[:-1] + self.cartesian_basis.shape[1:])

    def project_tensor(self, tensor):
        """Apply intrinsic index symmetry and the crystal's invariant projector."""
        return self.to_cartesian(self.from_cartesian(tensor) @ self.projector)

    def torch_metadata(self, dtype=torch.float32):
        """Fixed-size real readout tensors, suitable for mixed-space-group batches."""
        basis = torch.zeros(self.dim, self.dim, dtype=dtype)
        mask = torch.zeros(self.dim, dtype=dtype)
        basis[:, :self.invariant_dim] = torch.as_tensor(self.invariant_basis.copy(), dtype=dtype)
        mask[:self.invariant_dim] = 1
        return {
            "invariant_basis": basis,
            "invariant_coeff_mask": mask,
            "invariant_dim": self.invariant_dim,
            "symmetry_projector": torch.as_tensor(self.projector, dtype=dtype),
            "space_group_number": self.number,
            "sg_target": self.target,
        }

    def summary(self):
        multiplicities = {}
        for sector in self.sectors:
            key = str(sector.sg_irrep)
            item = multiplicities.setdefault(key, {
                "dimension": sector.irrep_dim, "multiplicity": 0, "trivial": sector.trivial
            })
            item["multiplicity"] += sector.multiplicity
        return {
            "target": self.target, "formula": self.formula, "o3_irreps": self.o3_irreps,
            "space_group_number": self.number, "international": self.international,
            "pointgroup": self.pointgroup, "kpoint": [0, 0, 0], "symprec": self.symprec,
            "tensor_dim": self.dim, "invariant_dim": self.invariant_dim,
            "num_operations": len(self.rotations), "sg_irreps": multiplicities,
            "branching": [asdict(sector) for sector in self.sectors],
            "irrep_label_convention": "local spgrep indices, not Bilbao/Mulliken labels",
        }


def decompose_property(structure, target, *, symprec=1e-5, kpoint=(0, 0, 0), tol=1e-5):
    """Branch each O(3) copy into SG irreps; no standardization of the tensor frame.

    This is for uniform, nonmagnetic, polar dielectric/piezoelectric/elastic
    tensors. Nonzero k requires a different property representation (including
    Bloch/site degrees of freedom) and is deliberately rejected.
    """
    kpoint = np.asarray(kpoint, dtype=float)
    if kpoint.shape != (3,) or not np.isfinite(kpoint).all():
        raise ValueError("kpoint must contain three finite numbers.")
    if np.any(kpoint != 0):
        raise NotImplementedError("Only Gamma (k=0) homogeneous tensors are supported.")
    if not np.isfinite(symprec) or symprec <= 0 or not 0 < tol < 0.01:
        raise ValueError("symprec must be positive and 0 < tol < 0.01.")
    try:
        from spgrep import get_crystallographic_pointgroup_irreps_from_symmetry
        from spgrep.representation import project_to_irrep
    except ImportError as exc:
        raise ImportError("Install requirements.txt in your environment.") from exc

    ct, q = tensor_basis(target)
    lattice, positions, numbers = as_cell(structure)
    dataset = spglib.get_symmetry_dataset((lattice, positions, numbers), symprec=symprec)
    if dataset is None:
        raise ValueError("spglib could not determine symmetry for this structure.")
    rotations, translations = dataset.rotations.copy(), dataset.translations.copy()
    # At Gamma every translation is in the kernel. Enumerate G/T and inflate
    # its irreps back onto ALL (R,t) operations, including centering copies.
    # This also avoids spgrep 0.3.8's get_spacegroup_irreps assumption that a
    # centered space group's input basis is already its conventional setting.
    unique_rotations, inverse = np.unique(rotations, axis=0, return_inverse=True)
    quotient_irreps = get_crystallographic_pointgroup_irreps_from_symmetry(unique_rotations)
    if sum(ir.shape[-1] ** 2 for ir in quotient_irreps) != len(unique_rotations):
        raise RuntimeError("spgrep did not enumerate a complete set of quotient irreps.")
    irreps = [irrep[inverse] for irrep in quotient_irreps]
    a = lattice.T
    cart_rotations = a @ rotations @ np.linalg.inv(a)
    if not np.allclose(cart_rotations @ cart_rotations.swapaxes(1, 2), np.eye(3), atol=tol, rtol=0):
        raise ValueError("Symmetry rotations are not orthogonal in the input frame; check symprec/structure.")
    representation = tensor_representation(q, cart_rotations)

    columns, sectors = [], []
    input_start, output_start = 0, 0
    # Keep every O(3) copy separate, so the report gives an explicit branching rule.
    for multiplicity, o3_irrep in ct:
        for copy in range(multiplicity):
            size = o3_irrep.dim
            block = representation[:, input_start:input_start + size, input_start:input_start + size]
            character = np.trace(block, axis1=1, axis2=2)
            for index, irrep in enumerate(irreps):
                ir_dim = irrep.shape[-1]
                ir_character = np.trace(irrep, axis1=1, axis2=2)
                count = np.vdot(ir_character, character) / len(rotations)
                count_int = int(round(count.real))
                if abs(count - count_int) > tol or count_int < 0:
                    raise ValueError("Noninteger irrep multiplicity; check symmetry tolerance and lattice.")
                if count_int == 0:
                    continue
                trivial = ir_dim == 1 and np.allclose(irrep, 1, atol=tol, rtol=0)
                candidates = project_to_irrep(block, irrep, atol=tol)
                if len(candidates) != count_int:
                    raise RuntimeError("spgrep did not recover all copies of an irrep.")
                copies = []
                for candidate in candidates:
                    u = candidate.T.copy()
                    # Orthogonalize equivalent copies without mixing rows of an irrep.
                    for previous in copies:
                        u -= previous * (np.trace(previous.conj().T @ u) / ir_dim)
                    norm = np.sqrt(np.vdot(u, u).real / ir_dim)
                    if norm < tol:
                        raise RuntimeError("Linearly dependent irrep copies.")
                    u /= norm
                    pivot = u.flat[np.abs(u).argmax()]
                    u *= np.exp(-1j * np.angle(pivot))
                    if trivial:
                        if np.abs(u.imag).max() > tol:
                            raise RuntimeError("Trivial irrep did not yield a real basis.")
                        u = u.real.astype(complex)
                    if not np.allclose(block @ u, u @ irrep, atol=tol, rtol=0):
                        raise RuntimeError("Irrep basis does not intertwine the tensor action.")
                    copies.append(u)
                    full = np.zeros((ct.dim, ir_dim), dtype=complex)
                    full[input_start:input_start + size] = u
                    columns.append(full)
                stop = output_start + count_int * ir_dim
                sectors.append(IrrepSector(str(o3_irrep), copy, index, ir_dim, count_int,
                                           output_start, stop, trivial))
                output_start = stop
            input_start += size
    basis = np.concatenate(columns, axis=1)
    if basis.shape != (ct.dim, ct.dim) or not np.allclose(basis.conj().T @ basis, np.eye(ct.dim), atol=tol, rtol=0):
        raise RuntimeError("Space-group decomposition is incomplete or not orthonormal.")
    result = SpaceGroupDecomposition(
        target, PROPERTY_FORMULAS[target], str(ct), int(dataset.number), dataset.international,
        dataset.pointgroup, symprec, rotations, translations, cart_rotations,
        dataset.equivalent_atoms.copy(), tuple(irreps), representation, basis, tuple(sectors), q,
    )
    if not np.allclose(result.projector, representation.mean(axis=0), atol=tol, rtol=0):
        raise RuntimeError("Trivial sectors disagree with the group-average projector.")
    return result
