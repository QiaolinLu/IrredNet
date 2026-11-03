"""Irrep gating and scatter utilities required by the standard backbone."""

from typing import Optional
import torch
from e3nn import o3


def tp_path_exists(irreps_in1, irreps_in2, ir_out) -> bool:
    """Check if the tensor product of irreps_in1 and irreps_in2 contains ir_out.
    
    Args:
        irreps_in1: Input irreps (e.g., "1x0e + 2x1e").
        irreps_in2: Input irreps (e.g., "3x0e").
        ir_out: Target irrep (e.g., "1e").
    
    Returns:
        bool: True if ir_out exists in irreps_in1 ⊗ irreps_in2.
    """
    irreps_in1 = o3.Irreps(irreps_in1).simplify()
    irreps_in2 = o3.Irreps(irreps_in2).simplify()
    ir_out = o3.Irrep(ir_out)  # Ensure ir_out is normalized
    for (_, ir1) in irreps_in1:
        for (_, ir2) in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False


def gated_irreps_layer_v2(
        tp_irreps_in1: o3.Irreps,
        tp_irreps_in2: o3.Irreps,
        tp_irreps_out: o3.Irreps,
        activation_type: str,
        activation_scalars: dict,
        activation_gates: dict,
        
):
    from e3nn.nn import Gate, NormActivation
    ir_tmp, _, _ = o3.Irreps(tp_irreps_out).sort()
    tp_irreps_out = ir_tmp.simplify()

    irreps_scalars = o3.Irreps(
        [
            (mul, ir)
            for mul, ir in tp_irreps_out
            if ir.l == 0 and tp_path_exists(tp_irreps_in1, tp_irreps_in2, ir)
        ]
    )
    irreps_gated = o3.Irreps(
        [
            (mul, ir)
            for mul, ir in tp_irreps_out
            if ir.l > 0 and tp_path_exists(tp_irreps_in1, tp_irreps_in2, ir)
        ]
    )
    if activation_type == "gate":
        # Setting all ir to 0e if there is path exist, since 0e gates will not
        # change the parity of the output, and we want to keep its parity.
        # If there is no 0e, we use 0o to change the party of the high-order irreps.
        if irreps_gated.dim > 0:
            if tp_path_exists(tp_irreps_in1, tp_irreps_in2, "0e"):
                ir = "0e"
            elif tp_path_exists(tp_irreps_in1, tp_irreps_in2, "0o"):
                ir = "0o"
            else:
                raise ValueError(
                    f"tp_irreps_in1={tp_irreps_in1} times "
                    f"tp_irreps_in2={tp_irreps_in2} is unable to produce gates "
                    f"needed for irreps_gated={irreps_gated}"
                )
        else:
            ir = None

        irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated]).simplify()

        activation = Gate(
            irreps_scalars=irreps_scalars,  # scalars
            act_scalars=[activation_scalars[ir.p] for _, ir in irreps_scalars],
            irreps_gates=irreps_gates,  # gates (scalars)
            act_gates=[activation_gates[ir.p] for _, ir in irreps_gates],
            irreps_gated=irreps_gated,  # gated tensors
        )
    
    elif activation_type == "norm":
        activation = NormActivation(
            irreps_in=(irreps_scalars + irreps_gated).simplify(),
            # norm is an even scalar, so activation_scalars[1]
            scalar_nonlinearity=activation_scalars[1],
            normalize=True,
            epsilon=1e-8,
            bias=False,
        )
    else:
        supported = ("gate", "norm")
        raise ValueError(
            f"Support `activation_type` includes {supported}, got {activation_type}"
        )
    return activation


def _broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand_as(other)
    return src


def scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
    reduce: str = "sum",
) -> torch.Tensor:
    assert reduce == "sum"  # for now, TODO
    index = _broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)
