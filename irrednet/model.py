"""IrredNet backbone with direct, variable-length independent-coefficient output."""

from types import SimpleNamespace

import torch
from e3nn import o3

from .backbone.model import IrredNet
from .coefficients import BASIS_VERSION, MODEL_VERSION, LABEL_CONVENTION, reconstruct_tensors, tensor_shape
from .decomposition import PROPERTY_FORMULAS, tensor_basis


class _PooledFeatures(torch.nn.Module):
    def forward(self, features, **unused):
        return features


class _CoefficientContraction(torch.nn.Module):
    """Contract matching (l, parity) copies directly onto each allowed basis row."""

    def __init__(self, feature_indices, basis_slice, input_copies, output_copies, dim):
        super().__init__()
        self.register_buffer("feature_indices", torch.tensor(feature_indices, dtype=torch.long))
        self.basis_slice = basis_slice
        self.input_copies, self.output_copies, self.dim = input_copies, output_copies, dim
        self.num_weights = input_copies * output_copies

    def forward(self, features, basis, weights, graph):
        h = features[:, self.feature_indices].reshape(-1, self.input_copies, self.dim)
        b = basis[:, self.basis_slice].reshape(-1, self.output_copies, self.dim)
        w = weights.reshape(-1, self.input_copies, self.output_copies)
        # Result has sum(r_i) entries, never a predicted full material-irrep vector.
        return torch.einsum("rcm,rtm,rct->r", h[graph], b, w[graph]) / self.input_copies ** .5


class SpaceGroupCoefficientHead(torch.nn.Module):
    """Equivariant feature/basis contractions with invariant, learned copy mixing.

    a_i = sum_(l,p,c,t,m) w_(l,p,c,t)(invariants) h_(l,p,c,m) B_(i,l,p,t,m).
    Nonlinear weights depend only on graph invariants, never on a basis row.
    This makes the output linear in B: B' = U B implies a' = U a, and hence
    the reconstructed physical tensor is independent of the basis gauge.
    """

    def __init__(self, target, irreps_in, hidden_dim=128, embedding_dim=16,
                 scalar_mode="contract", scalar_shift=0., scalar_scale=1.):
        super().__init__()
        if scalar_mode not in ("contract", "independent"):
            raise ValueError(f"Unknown scalar head mode {scalar_mode!r}")
        if scalar_mode == "independent" and target != "dielectric":
            raise ValueError("The independent scalar head is defined only for dielectric tensors")
        self.target = target
        self.scalar_mode = scalar_mode
        self.irreps_in = o3.Irreps(irreps_in)
        self.feature_dim = self.irreps_in.dim
        self.basis_dim = 3 ** len(tensor_shape(target))
        property_irreps, q = tensor_basis(target)
        self.register_buffer("cartesian_to_irreps", torch.tensor(q.reshape(property_irreps.dim, -1).T.copy()))
        self.paths = torch.nn.ModuleList()
        for (copies, irrep), basis_slice in zip(property_irreps, property_irreps.slices()):
            if scalar_mode == "independent" and irrep == o3.Irrep("0e"):
                if copies != 1:
                    raise ValueError("Expected exactly one dielectric 0e component")
                self.scalar_basis_slice = basis_slice
                continue
            indices = [i for (_, feature_irrep), part in zip(self.irreps_in, self.irreps_in.slices())
                       if feature_irrep == irrep for i in range(part.start, part.stop)]
            if not indices:
                raise ValueError(f"Backbone has no {irrep} channels required for {target}.")
            self.paths.append(_CoefficientContraction(indices, basis_slice, len(indices) // irrep.dim,
                                                       copies, irrep.dim))
        self.space_group_embedding = torch.nn.Embedding(231, embedding_dim)
        self.copy_weights = torch.nn.Sequential(
            torch.nn.Linear(self.irreps_in.num_irreps + embedding_dim, hidden_dim),
            torch.nn.SiLU(), torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.SiLU(), torch.nn.Linear(hidden_dim, sum(path.num_weights for path in self.paths)),
        )
        if scalar_mode == "independent":
            import math
            if not math.isfinite(scalar_shift) or not math.isfinite(scalar_scale) or scalar_scale <= 0:
                raise ValueError("Scalar initialization needs a finite shift and positive scale")
            self.register_buffer("scalar_shift", torch.tensor(float(scalar_shift)))
            self.register_buffer("scalar_scale", torch.tensor(float(scalar_scale)))
            self.scalar_readout = torch.nn.Sequential(
                torch.nn.Linear(self.irreps_in.num_irreps, hidden_dim), torch.nn.SiLU(),
                torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.SiLU(), torch.nn.Linear(hidden_dim, 1),
            )
            # Begin at the training-set median, with no validation/test statistics.
            torch.nn.init.zeros_(self.scalar_readout[-1].weight)
            torch.nn.init.zeros_(self.scalar_readout[-1].bias)

    def invariants(self, features):
        values = []
        for (copies, irrep), part in zip(self.irreps_in, self.irreps_in.slices()):
            block = features[:, part].reshape(-1, copies, irrep.dim)
            # Only 0e may enter a scalar MLP with its sign. In particular 0o
            # must be normed as well, otherwise reflection covariance is lost.
            values.append(block.squeeze(-1) if irrep == o3.Irrep("0e")
                          else (block.square().mean(-1) + 1e-8).sqrt())
        return torch.cat(values, dim=-1)

    def forward(self, features, data):
        if features.ndim != 2 or features.shape != (data.num_graphs, self.feature_dim):
            raise ValueError("Unexpected pooled backbone feature dimensions.")
        if data.canonical_basis.shape != (len(data.coefficient_batch), self.basis_dim):
            raise ValueError("Missing or invalid independent coefficient basis.")
        if data.coefficient_batch.numel() == 0:
            return (features.sum() * 0).expand(0)
        graph = data.coefficient_batch
        embedded = self.space_group_embedding(data.space_group_number)
        invariant_features = self.invariants(features)
        weights = self.copy_weights(torch.cat((invariant_features, embedded), dim=-1))
        basis = data.canonical_basis.to(features) @ self.cartesian_to_irreps.to(features)
        result = features.new_zeros(graph.numel())
        if self.scalar_mode == "independent":
            # This is a scalar 0e coefficient, not a predicted 3x3 tensor. Its
            # contraction with every allowed basis row preserves arbitrary
            # orthogonal changes of basis, including mixtures of 0e and 2e.
            scalar = self.scalar_shift + self.scalar_scale * self.scalar_readout(invariant_features).squeeze(-1)
            result = result + scalar[graph] * basis[:, self.scalar_basis_slice].squeeze(-1)
        offset = 0
        for path in self.paths:
            result = result + path(features, basis, weights[:, offset:offset + path.num_weights], graph)
            offset += path.num_weights
        return result


class IrredNetSpaceGroup(IrredNet):
    """Forward returns sum(r_i) coefficients; reconstruct returns input-frame tensors."""

    def __init__(self, args):
        target = args.target
        if target not in PROPERTY_FORMULAS or args.formula != PROPERTY_FORMULAS[target]:
            raise ValueError("Use the target's formula from PROPERTY_FORMULAS.")
        super().__init__(args)
        self.target = target
        # Remove the original material-irrep output layer. The inherited forward
        # now computes only embedding, interactions and mean pooling.
        self.TI = _PooledFeatures()
        self.coefficient_head = SpaceGroupCoefficientHead(
            target, args.hidden_irreps, getattr(args, "coefficient_hidden_dim", 128),
            scalar_mode=getattr(args, "dielectric_scalar_head", "contract"),
            scalar_shift=getattr(args, "dielectric_scalar_shift", 0.),
            scalar_scale=getattr(args, "dielectric_scalar_scale", 1.),
        )

    def forward(self, data):
        targets = getattr(data, "sg_target", None)
        versions = getattr(data, "coefficient_basis_version", None)
        if isinstance(targets, str):
            targets = [targets]
        if isinstance(versions, str):
            versions = [versions]
        if not targets or any(target != self.target for target in targets):
            raise ValueError(f"Expected coefficient metadata for {self.target}.")
        if not versions or any(version != BASIS_VERSION for version in versions):
            raise ValueError("Expected graphs in the versioned lattice coefficient frame.")
        conventions = getattr(data, "label_convention", None)
        if isinstance(conventions, str):
            conventions = [conventions]
        if not conventions or any(value != LABEL_CONVENTION for value in conventions):
            raise ValueError("Expected current JARVIS/VASP label convention metadata.")
        return self.coefficient_head(super().forward(data), data)

    def reconstruct(self, coefficients, data):
        return reconstruct_tensors(coefficients, data, self.target)

    def predict_tensor(self, data):
        return self.reconstruct(self(data), data)


def load_coefficient_model(checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if (checkpoint.get("version") != MODEL_VERSION or checkpoint.get("basis_version") != BASIS_VERSION
            or checkpoint.get("label_convention") != LABEL_CONVENTION):
        raise ValueError("Incompatible coefficient checkpoint; old MLP/projection models and "
                         "elastic labels in the old shear order must be retrained.")
    model = IrredNetSpaceGroup(SimpleNamespace(**checkpoint["model_args"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint
