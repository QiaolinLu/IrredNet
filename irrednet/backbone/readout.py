"""Original tensor readout retained for constructor/RNG compatibility.

IrredNetSpaceGroup replaces this readout with pooled features before prediction;
all actual material outputs come from the independent coefficient head.
"""

from typing import Optional
import torch
from e3nn import o3
from e3nn.io import CartesianTensor


def apply_tensor_symmetry_projector(
    features: torch.Tensor,
    symmetry_projector: Optional[torch.Tensor],
) -> torch.Tensor:
    """Project tensor irreps coefficients onto the point-group invariant subspace."""
    if symmetry_projector is None:
        return features

    squeeze_back = False
    if features.ndim == 1:
        features = features.unsqueeze(0)
        squeeze_back = True
    if features.ndim != 2:
        raise ValueError(
            f"Expected features with shape [batch, dim] or [dim], got {tuple(features.shape)}."
        )

    projector = symmetry_projector.to(device=features.device, dtype=features.dtype)
    if projector.ndim == 2:
        projected = torch.einsum("ij,bj->bi", projector, features)
    elif projector.ndim == 3:
        if projector.shape[0] != features.shape[0]:
            raise ValueError(
                "Batched symmetry projector must have the same batch dimension as features: "
                f"{projector.shape[0]} != {features.shape[0]}."
            )
        projected = torch.einsum("bij,bj->bi", projector, features)
    else:
        raise ValueError(
            f"Expected symmetry projector with shape [dim, dim] or [batch, dim, dim], "
            f"got {tuple(projector.shape)}."
        )

    if squeeze_back:
        projected = projected.squeeze(0)
    return projected


def apply_invariant_basis_decomposition(
    features: torch.Tensor,
    invariant_basis: Optional[torch.Tensor],
    invariant_coeff_mask: Optional[torch.Tensor] = None,
):
    """Express a tensor in the point-group invariant basis and reconstruct it."""
    if invariant_basis is None:
        return features, None

    squeeze_back = False
    if features.ndim == 1:
        features = features.unsqueeze(0)
        squeeze_back = True
    if features.ndim != 2:
        raise ValueError(
            f"Expected features with shape [batch, dim] or [dim], got {tuple(features.shape)}."
        )

    basis = invariant_basis.to(device=features.device, dtype=features.dtype)
    if basis.ndim == 2:
        basis = basis.unsqueeze(0).expand(features.shape[0], -1, -1)
    elif basis.ndim != 3:
        raise ValueError(
            f"Expected invariant basis with shape [dim, dim] or [batch, dim, dim], got {tuple(basis.shape)}."
        )

    coeff_mask = invariant_coeff_mask
    if coeff_mask is None:
        coeff_mask = torch.ones(
            basis.shape[0],
            basis.shape[-1],
            device=features.device,
            dtype=features.dtype,
        )
    else:
        coeff_mask = coeff_mask.to(device=features.device, dtype=features.dtype)
        if coeff_mask.ndim == 1:
            coeff_mask = coeff_mask.unsqueeze(0).expand(features.shape[0], -1)
        elif coeff_mask.ndim != 2:
            raise ValueError(
                f"Expected invariant coefficient mask with shape [dim] or [batch, dim], got {tuple(coeff_mask.shape)}."
            )

    coeffs = torch.einsum("bdi,bd->bi", basis, features)
    coeffs = coeffs * coeff_mask
    reconstructed = torch.einsum("bdi,bi->bd", basis, coeffs)

    if squeeze_back:
        reconstructed = reconstructed.squeeze(0)
        coeffs = coeffs.squeeze(0)

    return reconstructed, coeffs


class FastTensorIrreps(torch.nn.Module):
    def __init__(self ,formula , irreps_in):
        super().__init__()
        if formula is None:
            self.formula=formula
            self.irreps_in = irreps_in
            self.irreps_out = o3.Irreps('0e')
            self.extra_layers = o3.Linear(irreps_in=self.irreps_in, irreps_out=self.irreps_out)
        else:
            self.formula=formula
            self.irreps_in = irreps_in
            self.irreps_out = CartesianTensor(formula=self.formula)
            lhs = self.formula.split('=')[0]
            index_rank = len(lhs)
            # Build dummy tensor with just the Cartesian indices
            dummy_shape = tuple(3 for _ in range(index_rank))  # e.g. (3, 3) for "ij"
            dummy = torch.empty(*dummy_shape, dtype=torch.float32)
            # Build and cache the change-of-basis matrix !!! important !!!
            self.rtp = self.irreps_out.reduced_tensor_products(dummy)
            self.extra_layers = o3.Linear(irreps_in=self.irreps_in, irreps_out=self.irreps_out)

    def forward(
        self,
        data,
        symmetry_projector: Optional[torch.Tensor] = None,
        invariant_basis: Optional[torch.Tensor] = None,
        invariant_coeff_mask: Optional[torch.Tensor] = None,
    ):
        out=self.extra_layers(data)
        if invariant_basis is not None:
            out, _ = apply_invariant_basis_decomposition(
                out,
                invariant_basis,
                invariant_coeff_mask=invariant_coeff_mask,
            )
        else:
            out = apply_tensor_symmetry_projector(out, symmetry_projector)
        if self.formula is None:
            return out
        else:
            out = self.irreps_out.to_cartesian(out, rtp=self.rtp)
            return out
