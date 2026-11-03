# Method and implementation

[Back to the project](../README.md)

## Symmetry-allowed tensor coordinates

The intrinsic index symmetries of a material property define a Cartesian tensor
space. e3nn provides an orthonormal change of basis into O(3) irreducible
representations. IrredNet restricts this representation to the symmetry
operations of each crystal.

For a structure with symmetry operations `g`, the group-average projector is

```text
P = (1 / |G|) Σ_g D(g)
```

Its invariant subspace contains the tensors allowed by the crystal symmetry.
Orthonormalizing this subspace produces a basis `B₁, …, Bᵣ`. The model predicts
`r` independent coefficients and reconstructs

```text
T = Σₖ aₖ Bₖ
```

The tensor has the required intrinsic index symmetries and lies in the detected
group's allowed space. Examples of the allowed dimensions are:

| Space group | Dielectric | Piezoelectric | Elastic |
|---|---:|---:|---:|
| `P1` | 6 | 18 | 21 |
| `P-1` | 6 | 0 | 21 |
| `P3` | 2 | 6 | 7 |
| `P41` | 2 | 4 | 7 |
| `P4mm` | 2 | 3 | 6 |
| `P6/mmm` | 2 | 0 | 5 |
| `F-43m` | 1 | 1 | 3 |
| `Pm-3m` | 1 | 0 | 3 |

The allowed dimension depends on the actual group action, not only on the
crystal-system name. Complete irrep branching is available through
`decompose_property`; coefficient prediction uses its invariant sector.

## Equivariant coefficient prediction

The standard IrredNet backbone embeds the periodic graph and processes it using
spherical harmonics, radial functions, and gated tensor-product interactions.
After graph pooling, the coefficient head contracts matching `(l, parity)`
feature channels with each allowed basis tensor. Invariant features control the
learned mixing of irrep copies; a space-group embedding conditions this mixing.

The head is linear in the allowed basis. Under an orthogonal basis change
`B′ = U B`, coefficients transform as `a′ = U a`, leaving `a′ᵀ B′ = aᵀ B`
unchanged. The optional dielectric scalar branch preserves this property.

Different structures can have different `r`. Coefficients are packed into one
vector with offsets stored in `coefficient_ptr`. No padded coefficients are
included in the loss. A symmetry-forbidden tensor has `r = 0` and reconstructs
directly to zero.

## Coordinate frames

A QR factorization of the lattice defines an orthogonal canonical frame.
Reflections are retained. Graph features and the canonical basis use this frame;
the reconstruction basis is rotated back to the input Cartesian frame.

For fixed ordered lattice vectors, rigid rotations and reflections preserve the
predicted coefficients and transform the reconstructed tensor covariantly.
Changing the lattice basis can change coefficient coordinates; the reconstructed
physical tensor is the object to compare.

## Physical scope and numerical precision

This implementation treats **homogeneous, nonmagnetic polar tensors at Γ**.
Translations act trivially on these properties, including fractional translations
in nonsymmorphic space groups. Space-group constraints therefore use the point-
group action. Finite-wavevector responses, magnetic groups, and axial tensors
require different representations and are not implemented here.

The basis construction enforces symmetry mathematically. Numerical residuals
depend on floating-point precision, the structure, and the `symprec` used for
symmetry identification. Exact agreement at every absolute numerical tolerance
is not implied.

## Model compatibility

The implementation uses the standard interaction path and the following
checkpoint conventions:

| Field | Value |
|---|---|
| Model | `space_group_equivariant_coefficients_v3` |
| Basis | `lattice_qr_o3_projector_mgs_v2` |
| Labels | `jarvis_vasp_unscaled_xx_yy_zz_xy_yz_xz_v1` |

The loader checks these conventions before loading weights. Earlier MLP heads,
different compressed tensor orders, and experimental SO2 checkpoints are not
compatible with this implementation.

The original backbone constructor and parameter names are retained for weight
and initialization compatibility. Its tensor readout is constructed and then
replaced by pooled features before the coefficient head is created.

In the current standard backbone, `NormalizationLayer.forward` leaves its
LayerNorm branch inactive, while the layer's parameters remain in checkpoints.
Enabling that branch would change training behavior and requires a separate
model revision.
