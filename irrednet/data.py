"""Read-only input loading and fresh per-structure SG metadata for the new version."""

from functools import lru_cache
import pickle
import time

import numpy as np
import torch

from .decomposition import PROPERTY_FORMULAS, as_cell
from .coefficients import (
    BASIS_VERSION, LABEL_CONVENTION, ELASTIC_PAIRS, PIEZO_PAIRS,
    build_coefficient_basis, labels_to_cartesian,
)



class _CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "spglib.spglib":
            module = "spglib"
        return super().find_class(module, name)


def load_samples(path):
    """Load existing trusted project pickles without rewriting their caches."""
    with open(path, "rb") as handle:
        samples = _CompatUnpickler(handle).load()
    if not isinstance(samples, (list, tuple)):
        raise ValueError("Expected a list of preprocessed sample dictionaries.")
    return samples


def sample_structure(sample):
    # p_input is the frame used by the existing graph builder and tensor labels.
    candidates = (sample.get("p_input") or {}, sample)
    for candidate in candidates:
        structure = candidate.get("structure")
        if structure is not None:
            return structure
    if sample.get("corrected_structure") is not None:
        return sample["corrected_structure"]
    if sample.get("atoms") is not None:
        from jarvis.core.atoms import Atoms
        from pymatgen.io.jarvis import JarvisAtomsAdaptor
        return JarvisAtomsAdaptor.get_structure(Atoms.from_dict(sample["atoms"]))
    raise ValueError("Sample has no structure in the target tensor's coordinate frame.")


@lru_cache(maxsize=1)
def element_features():
    """Magpie/one-hot atom features, preserving the original IrredNet encoder."""
    from ase.atoms import Atom
    from jarvis.core.specie import Specie
    from sklearn.preprocessing import OneHotEncoder
    descriptors = [Specie(Atom(z).symbol, source="magpie").get_descrp_arr for z in range(1, 102)]
    try:
        encoder = OneHotEncoder(max_categories=6, sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(max_categories=6, sparse=False)
    return torch.as_tensor(encoder.fit_transform(descriptors), dtype=torch.float32)


def structure_graph(structure, cutoff):
    import ase.neighborlist
    from pymatgen.io.ase import AseAtomsAdaptor
    from torch_geometric.data import Data
    atoms = AseAtomsAdaptor.get_atoms(structure)
    src, dst, shift = ase.neighborlist.neighbor_list("ijS", atoms, cutoff, self_interaction=False)
    numbers = np.asarray(atoms.numbers)
    if np.any(numbers < 1) or np.any(numbers > 101):
        raise ValueError("The existing atom feature encoder supports atomic numbers 1..101.")
    return Data(
        x=element_features()[torch.as_tensor(numbers - 1)].clone(),
        edge_index=torch.as_tensor(np.stack((src, dst)), dtype=torch.long),
        edge_shift=torch.as_tensor(shift, dtype=torch.float32),
        pos=torch.as_tensor(atoms.positions, dtype=torch.float32),
        lattice=torch.as_tensor(atoms.cell.array, dtype=torch.float32).unsqueeze(0),
    )


class SpaceGroupDataset(torch.utils.data.Dataset):
    """Lattice-frame graphs and packed independent coefficients.

    Basis tensors for reconstruction are in the original Cartesian frame.
    Set require_labels=False for structures-only inference. No input is mutated.
    """

    def __init__(self, samples, target, *, symprec=1e-5, cutoff=5.0, require_labels=True,
                 progress_every=0):
        from pymatgen.core import Structure
        if target not in PROPERTY_FORMULAS:
            raise ValueError(f"Unknown target {target!r}.")
        if not np.isfinite(cutoff) or cutoff <= 0:
            raise ValueError("cutoff must be positive.")
        self.target, self.graphs, self.basis_summaries = target, [], []
        started = time.monotonic()
        for index, sample in enumerate(samples):
            if require_labels and target not in sample:
                raise ValueError(f"Sample {index} lacks {target!r}; supply a preprocessed dataset.")
            structure = sample_structure(sample)
            basis = build_coefficient_basis(structure, target, symprec=symprec)
            lattice, positions, numbers = as_cell(structure)
            canonical_structure = Structure(basis.canonical_lattice, numbers, positions)
            graph = structure_graph(canonical_structure, cutoff)
            # Isolated atoms in large cells legitimately have no neighbors.
            # The backbone retains their on-site features through its skip path.
            graph.canonical_basis = torch.tensor(basis.canonical_basis, dtype=torch.float32)
            graph.cartesian_basis = torch.tensor(basis.cartesian_basis, dtype=torch.float32)
            graph.coefficient_frame = torch.tensor(basis.frame, dtype=torch.float32).unsqueeze(0)
            graph.input_lattice = torch.tensor(lattice, dtype=torch.float32).unsqueeze(0)
            graph.num_coefficients = basis.dim
            graph.space_group_number = basis.symmetry["space_group_number"]
            graph.sg_target = target
            graph.coefficient_basis_version = BASIS_VERSION
            graph.label_convention = LABEL_CONVENTION
            graph.sample_id = str(sample.get("JARVIS_ID", sample.get("jid", sample.get("id", index))))
            if require_labels:
                labels = torch.tensor(np.asarray(sample[target]), dtype=torch.float32)
                if not torch.isfinite(labels).all():
                    raise ValueError(f"Sample {index} contains nonfinite labels.")
                cartesian = labels_to_cartesian(labels, target)
                graph.labels = labels
                graph.cartesian_labels = cartesian.reshape(1, -1)
                graph.coefficient_labels = graph.cartesian_basis @ cartesian.flatten()
                projected = graph.coefficient_labels @ graph.cartesian_basis
                graph.label_projection_residual = (projected - cartesian.flatten()).norm().reshape(1)
            self.graphs.append(graph)
            # Do not retain all group representation matrices for every training
            # sample; only the small basis tensors and summaries are needed.
            self.basis_summaries.append(basis.symmetry)
            if progress_every and (index == 0 or (index + 1) % progress_every == 0
                                   or index + 1 == len(samples)):
                print(f"Prepared {index + 1}/{len(samples)} structures "
                      f"({time.monotonic() - started:.1f}s)", flush=True)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        return self.graphs[index].clone()

    @staticmethod
    def collate(graphs):
        from torch_geometric.data import Batch
        if not graphs:
            raise ValueError("Cannot collate an empty graph list.")
        targets = {graph.sg_target for graph in graphs}
        if len(targets) != 1:
            raise ValueError("A batch must contain a single tensor property.")
        labeled = ["labels" in graph for graph in graphs]
        if any(labeled) and not all(labeled):
            raise ValueError("Do not mix labeled and unlabeled samples in a batch.")
        batch = Batch.from_data_list(graphs, exclude_keys=["labels"])
        if all(labeled):
            batch.labels = torch.stack([graph.labels for graph in graphs])
        counts = batch.num_coefficients
        batch.coefficient_ptr = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
        batch.coefficient_batch = torch.repeat_interleave(torch.arange(len(graphs)), counts)
        src, dst = batch.edge_index
        batch.edge_vec = batch.pos[dst] - batch.pos[src] + torch.einsum(
            "ni,nij->nj", batch.edge_shift, batch.lattice[batch.batch[src]]
        )
        return batch


def match_label_shape(prediction, labels, target):
    """Match JARVIS/VASP orders, with no engineering/Mandel scaling."""
    if prediction.shape == labels.shape:
        return prediction
    if target == "piezoelectric" and labels.shape[1:] == (3, 6):
        j, k = zip(*PIEZO_PAIRS)
        return prediction[:, :, list(j), list(k)]
    if target == "elastic" and labels.shape[1:] == (6, 6):
        i, j = zip(*ELASTIC_PAIRS)
        return prediction[:, list(i), list(j)][:, :, list(i), list(j)]
    raise ValueError(f"Cannot match {target} prediction {prediction.shape} to labels {labels.shape}.")
