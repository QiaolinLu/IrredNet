"""IrredNet backbone blocks, extracted without changing numerical behavior."""

from abc import abstractmethod
from typing import List, Optional, Dict
import math
import torch
import torch.nn.functional as F
from e3nn import nn, o3
from e3nn.nn import BatchNorm
from e3nn.util.jit import compile_mode
from torch_scatter import scatter
from .radial import (BesselBasis, GaussianBasis, ChebychevBasis,
                     AgnesiTransform, SoftTransform, PolynomialCutoff)
from .irreps_tools import tp_out_irreps_with_instructions
from .utils import scatter_sum, gated_irreps_layer_v2


class LinearNodeEmbeddingBlock(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        shared_weights: bool = True,
        internal_weights: bool = True,
    ):
        super().__init__()
        self.linear = o3.Linear(
            irreps_in=irreps_in, 
            irreps_out=irreps_out, 
            shared_weights=shared_weights,
            internal_weights=internal_weights
        )

    def forward(
        self,
        node_attrs: torch.Tensor,
    ) -> torch.Tensor:  # [n_nodes, irreps]
        return self.linear(node_attrs)


class RadialEmbeddingBlock(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        radial_type: str = "bessel",
        distance_transform: str = "None",
    ):
        super().__init__()
        if radial_type == "bessel":
            self.bessel_fn = BesselBasis(r_max=r_max, num_basis=num_bessel)
        elif radial_type == "gaussian":
            self.bessel_fn = GaussianBasis(r_max=r_max, num_basis=num_bessel)
        elif radial_type == "chebyshev":
            self.bessel_fn = ChebychevBasis(r_max=r_max, num_basis=num_bessel)
        if distance_transform == "Agnesi":
            self.distance_transform = AgnesiTransform()
        elif distance_transform == "Soft":
            self.distance_transform = SoftTransform()
        self.cutoff_fn = PolynomialCutoff(r_max=r_max, p=num_polynomial_cutoff)
        self.out_dim = num_bessel
        
    def forward(
        self,
        edge_lengths: torch.Tensor,  # [n_edges, 1]
        node_attrs: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        atomic_numbers: Optional[torch.Tensor] = None,
    ):
        cutoff = self.cutoff_fn(edge_lengths)  # [n_edges, 1]
        if hasattr(self, "distance_transform") \
            and atomic_numbers is not None:
            edge_lengths = self.distance_transform(
                edge_lengths, node_attrs, edge_index, atomic_numbers
            )
        radial = self.bessel_fn(edge_lengths)  # [n_edges, n_basis]
        return radial * cutoff  # [n_edges, n_basis]


class InteractionBlock(torch.nn.Module):
    def __init__(
        self,
        node_attrs_irreps: o3.Irreps,
        node_feats_irreps: o3.Irreps,
        edge_attrs_irreps: o3.Irreps,
        edge_feats_irreps: o3.Irreps,
        target_irreps: o3.Irreps,
        hidden_irreps: Optional[o3.Irreps] = None,
        avg_num_neighbors: Optional[float] = None,
        radial_MLP: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        self.node_attrs_irreps = node_attrs_irreps
        self.node_feats_irreps = node_feats_irreps
        self.edge_attrs_irreps = edge_attrs_irreps
        self.edge_feats_irreps = edge_feats_irreps
        self.target_irreps = target_irreps
        self.hidden_irreps = hidden_irreps
        self.avg_num_neighbors = avg_num_neighbors
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
        self.radial_MLP = radial_MLP

        self._setup()

    @abstractmethod
    def _setup(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError


@compile_mode("script")
class RealAgnosticTensorInteractionBlock(InteractionBlock):
    def _setup(self) -> None:
        # First Linear 
        self.lin1 = o3.FullyConnectedTensorProduct(
            self.node_feats_irreps, 
            self.node_attrs_irreps,
            self.node_feats_irreps,
        )
        # Tensor Product
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            irreps1 = self.node_feats_irreps,
            irreps2 = self.edge_attrs_irreps,
            target_irreps = self.target_irreps,
        )
        self.conv_tp = o3.TensorProduct(
            irreps_in1 = self.node_feats_irreps,
            irreps_in2 = self.edge_attrs_irreps,
            irreps_out = irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )
        # Convolution weights using edge length embedding
        input_dim = self.edge_feats_irreps.num_irreps
        self.conv_tp_weights = nn.FullyConnectedNet(
            [input_dim] + self.radial_MLP + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,
        )
        tp_irreps_out = irreps_mid.simplify()

        # Second Linear 
        self.lin2 = o3.FullyConnectedTensorProduct(
            tp_irreps_out, 
            self.node_attrs_irreps, 
            self.target_irreps,
        )
        self.sc = o3.FullyConnectedTensorProduct(
            self.node_feats_irreps,
            self.node_attrs_irreps,
            self.target_irreps,
        )
    

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor, # spherical harmonic
        edge_feats: torch.Tensor, # radial embedding
        edge_index: torch.Tensor,
    ):
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]
        node_feats = self.lin1(node_feats, node_attrs)
        tp_weights = self.conv_tp_weights(edge_feats)
        mji = self.conv_tp(
            node_feats[sender], 
            edge_attrs,
            tp_weights,
        )
        message = scatter_sum(
            src=mji,
            index=receiver,
            dim=0,
            dim_size=num_nodes,
        )
        # aggregated_message = message.div(self.avg_num_neighbors**0.5)
        # node_out = self.lin2(message, node_attrs)  / self.avg_num_neighbors
        node_out = self.lin2(message, node_attrs)  
        node_sc = self.sc(node_feats, node_attrs)
        node_feats = node_out + node_sc 
        return node_feats
    
    @property
    def irreps_out(self):
        return self.sc.irreps_out


class ShiftedSoftPlus(torch.nn.Module):
    """
    Shifted softplus as defined in SchNet, NeurIPS 2017.

    :param beta: value for the a more general softplus, default = 1
    :param threshold: values above are linear function, default = 20
    """

    _log2: float

    def __init__(self, beta=1, threshold=20):
        super().__init__()
        self.softplus = torch.nn.Softplus(beta=beta, threshold=threshold)
        self._log2 = math.log(2.0)

    def forward(self, x):
        """
        Evaluate shifted softplus

        :param x: torch.Tensor, input
        :return: torch.Tensor, ssp(x)
        """
        return self.softplus(x) - self._log2


ACTIVATION = {
    # for even irreps
    "e": {
        "ssp": ShiftedSoftPlus(),
        "silu": torch.nn.functional.silu,
        "sigmoid": torch.sigmoid,
    },
    # for odd irreps
    "o": {
        "abs": torch.abs,
        "tanh": torch.tanh,
    },
}


@compile_mode("script")
class GateLayer(torch.nn.Module):
    def __init__(
            self, 
            tp_irreps_in1: o3.Irreps,
            tp_irreps_in2: o3.Irreps,
            tp_irreps_out: o3.Irreps,
            activation_type: str = "gate",
            activation_scalars: Optional[Dict[str, str]] = None,
            activation_gates: Optional[Dict[str, str]] = None,
    ):
        super().__init__()
        # activation function for even (i.e. 1) and odd (i.e. -1) scalars
        if activation_scalars is None:
            activation_scalars = {
                1: ACTIVATION["e"]["ssp"],
                # odd scalars requires either an even or odd activation,
                # not an arbitrary one like relu
                -1: ACTIVATION["o"]["tanh"],
            }
        else:
            # change key from e or v to 1 or -1
            key_mapping = {"e": 1, "o": -1}
            activation_scalars = {
                key_mapping[k]: ACTIVATION[k][v] for k, v in activation_scalars.items()
            }
        # activation function for even (i.e. 1) and odd (i.e. -1) high-order tensors
        if activation_gates is None:
            activation_gates = {1: ACTIVATION["e"]["ssp"], -1: ACTIVATION["o"]["abs"]}
        else:
            # change key from e or v to 1 or -1
            key_mapping = {"e": 1, "o": -1}
            activation_gates = {
                key_mapping[k]: ACTIVATION[k][v] for k, v in activation_gates.items()
            }

        self.activation = gated_irreps_layer_v2(
            tp_irreps_in1=tp_irreps_in1,
            tp_irreps_in2=tp_irreps_in2,
            tp_irreps_out=tp_irreps_out,
            activation_type=activation_type,
            activation_scalars=activation_scalars,
            activation_gates=activation_gates,
        )

    def forward(self, node_feats:torch.Tensor) -> torch.Tensor:
        return self.activation(node_feats)
    
    @property
    def irreps_in(self):
        return self.activation.irreps_in

    @property
    def irreps_out(self):
        return self.activation.irreps_out


class NormalizationLayer(torch.nn.Module):
    def __init__(
        self, 
        irreps: o3.Irreps,
        norm_type: str = None,
    ):
        super().__init__()
        self.norm_type = norm_type
        if self.norm_type == "BatchNorm":
            self.norm = BatchNorm(irreps, eps=1e-5)
        elif self.norm_type == "InstanceNorm":
            self.norm = InstanceNorm(irreps, eps=1e-5)
        elif self.norm_type == "LayerNorm":
            self.norm = LayerNorm(irreps, eps=1e-5)
        else:
            raise ValueError(f"Unknown normalization type: {norm_type}")
    
    def forward(self, 
                node_feats: torch.Tensor,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.norm_type == "BatchNorm":
            node_feats = self.norm(node_feats)
        elif self.norm_type == "InstanceNorm":
            node_feats = self.norm(node_feats, batch)
        return node_feats


class InstanceNorm(torch.nn.Module):
    """
    Instance normalization for orthonormal representations
    It normalizes by the norm of the representations.
    Note that the norm is invariant only for orthonormal representations.
    Irreducible representations `wigner_D` are orthonormal.
    Parameters
    ----------
    irreps : `Irreps`
        representation
    eps : float
        avoid division by zero when we normalize by the variance
    affine : bool
        do we have weight and bias parameters
    reduce : {'mean', 'max'}
        method used to reduce
    """

    def __init__(
        self, irreps, eps=1e-5, affine=True, reduce="mean", normalization="component"
    ):
        super().__init__()

        self.irreps = o3.Irreps(irreps)
        self.eps = eps
        self.affine = affine

        num_scalar = sum(mul for mul, ir in self.irreps if ir.l == 0)
        num_features = self.irreps.num_irreps

        if affine:
            self.weight = torch.nn.Parameter(torch.ones(num_features))
            self.bias = torch.nn.Parameter(torch.zeros(num_scalar))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        # TODO, this does not work for saving data.
        # mean and stdev, and norm should be buffer such that we save the model and
        # load back, they will be recovered, Take a look at the e3nn BatchNorm
        # implementation

        assert isinstance(reduce, str), "reduce should be passed as a string value"
        assert reduce in ["mean", "max"], "reduce needs to be 'mean' or 'max'"
        self.reduce = reduce

        assert normalization in [
            "norm",
            "component",
        ], "normalization needs to be 'norm' or 'component'"
        self.normalization = normalization

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.irreps}, eps={self.eps})"

    def forward(self, input, batch):
        """evaluate
        Parameters
        ----------
        input : `torch.Tensor`
            tensor of shape ``(batch, ..., irreps.dim)``
        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(batch, ..., irreps.dim)``
        """
        # batch, *size, dim = input.shape  # TODO: deal with batch
        # input = input.reshape(batch, -1, dim)  # [batch, sample, stacked features]
        # input has shape [batch * nodes, dim], but with variable nr of nodes.
        # the input batch slices this into separate graphs
        dim = input.shape[-1]

        fields = []
        ix = 0
        iw = 0
        ib = 0

        for mul, ir in self.irreps:
            # mul is the multiplicity (number of copies) of some irrep type (ir)
            d = ir.dim
            field = input[:, ix : ix + mul * d]  # [batch * sample, mul * repr]
            ix += mul * d

            # [batch * sample, mul, repr]
            field = field.reshape(-1, mul, d)

            # For scalars first compute and subtract the mean
            if ir.l == 0:
                # Compute the mean
                field_mean = global_mean_pool(field, batch).reshape(
                    -1, mul, 1
                )  # [batch, mul, 1]]
                # Subtract the mean
                field = field - field_mean[batch]

            # Then compute the rescaling factor (norm of each feature vector)
            # Rescaling of the norms themselves based on the option "normalization"
            if self.normalization == "norm":
                field_norm = field.pow(2).sum(-1)  # [batch * sample, mul]
            elif self.normalization == "component":
                field_norm = field.pow(2).mean(-1)  # [batch * sample, mul]
            else:
                raise ValueError(
                    "Invalid normalization option {}".format(self.normalization)
                )
            # Reduction method
            if self.reduce == "mean":
                field_norm = global_mean_pool(field_norm, batch)  # [batch, mul]
            elif self.reduce == "max":
                field_norm = global_max_pool(field_norm, batch)  # [batch, mul]
            else:
                raise ValueError("Invalid reduce option {}".format(self.reduce))

            # Then apply the rescaling (divide by the sqrt of the squared_norm,
            # i.e., divide by the norm
            field_norm = (field_norm + self.eps).pow(-0.5)  # [batch, mul]

            if self.affine:
                weight = self.weight[None, iw : iw + mul]  # [batch, mul]
                iw += mul
                field_norm = field_norm * weight  # [batch, mul]

            field = field * field_norm[batch].reshape(
                -1, mul, 1
            )  # [batch * sample, mul, repr]

            if self.affine and d == 1:  # scalars
                bias = self.bias[ib : ib + mul]  # [batch, mul]
                ib += mul
                field += bias.reshape(mul, 1)  # [batch * sample, mul, repr]

            # Save the result, to be stacked later with the rest
            fields.append(field.reshape(-1, mul * d))  # [batch * sample, mul * repr]

        if ix != dim:
            fmt = "`ix` should have reached input.size(-1) ({}), but it ended at {}"
            msg = fmt.format(dim, ix)
            raise AssertionError(msg)

        output = torch.cat(fields, dim=-1)  # [batch * sample, stacked features]
        return output


class LayerNorm(torch.nn.Module):

    def __init__(self, irreps, eps=1e-5, affine=True, normalization='component'):
        super().__init__()

        self.irreps = o3.Irreps(irreps)
        self.eps = eps
        self.affine = affine

        num_scalar = sum(mul for mul, ir in self.irreps if ir.l == 0 and ir.p == 1)
        num_features = self.irreps.num_irreps

        if affine:
            self.affine_weight = torch.nn.Parameter(torch.ones(num_features))
            self.affine_bias = torch.nn.Parameter(torch.zeros(num_scalar))
        else:
            self.register_parameter('affine_weight', None)
            self.register_parameter('affine_bias', None)

        assert normalization in ['norm', 'component'], "normalization needs to be 'norm' or 'component'"
        self.normalization = normalization

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.irreps}, eps={self.eps})"

    def forward(self, node_input, **kwargs):
        '''
            Use torch layer norm for scalar features.
        '''

        dim = node_input.shape[-1]

        fields = []
        ix = 0
        iw = 0
        ib = 0

        for mul, ir in self.irreps:  # mul is the multiplicity (number of copies) of some irrep type (ir)
            d = ir.dim
            field = node_input.narrow(1, ix, mul * d)
            ix += mul * d

            if ir.l == 0 and ir.p == 1:
                weight = self.affine_weight[iw:(iw + mul)]
                bias = self.affine_bias[ib:(ib + mul)]
                iw += mul
                ib += mul
                field = F.layer_norm(field, tuple((mul,)), weight, bias, self.eps)
                fields.append(field.reshape(-1, mul * d))  # [batch * sample, mul * repr]
                continue

            # For non-scalar features, use RMS value for std
            field = field.reshape(-1, mul, d)  # [batch * sample, mul, repr]

            if self.normalization == 'norm':
                field_norm = field.pow(2).sum(-1)  # [batch * sample, mul]
            elif self.normalization == 'component':
                field_norm = field.pow(2).mean(-1)  # [batch * sample, mul]
            else:
                raise ValueError("Invalid normalization option {}".format(self.normalization))
            field_norm = torch.mean(field_norm, dim=1, keepdim=True)
            field_norm = 1.0 / ((field_norm + self.eps).sqrt())  # [batch * sample, mul]

            if self.affine:
                weight = self.affine_weight[None, iw:(iw + mul)]  # [1, mul]
                iw += mul
                field_norm = field_norm * weight  # [batch * sample, mul]
            field = field * field_norm.reshape(-1, mul, 1)  # [batch * sample, mul, repr]
            fields.append(field.reshape(-1, mul * d))  # [batch * sample, mul * repr]

        assert ix == dim

        output = torch.cat(fields, dim=-1)
        return output


def global_mean_pool(x, batch, size: Optional[int] = None):
    r"""Returns batch-wise graph-level-outputs by averaging node features
    across the node dimension, so that for a single graph
    :math:`\mathcal{G}_i` its output is computed by

    .. math::
        \mathbf{r}_i = \frac{1}{N_i} \sum_{n=1}^{N_i} \mathbf{x}_n

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch (LongTensor): Batch vector :math:`\mathbf{b} \in {\{ 0, \ldots,
            B-1\}}^N`, which assigns each node to a specific example.
        size (int, optional): Batch-size :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """

    size = int(batch.max().item() + 1) if size is None else size
    return scatter(x, batch, dim=0, dim_size=size, reduce="mean")


def global_max_pool(x, batch, size: Optional[int] = None):
    r"""Returns batch-wise graph-level-outputs by taking the channel-wise
    maximum across the node dimension, so that for a single graph
    :math:`\mathcal{G}_i` its output is computed by

    .. math::
        \mathbf{r}_i = \mathrm{max}_{n=1}^{N_i} \, \mathbf{x}_n

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch (LongTensor): Batch vector :math:`\mathbf{b} \in {\{ 0, \ldots,
            B-1\}}^N`, which assigns each node to a specific example.
        size (int, optional): Batch-size :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """
    size = int(batch.max().item() + 1) if size is None else size
    return scatter(x, batch, dim=0, dim_size=size, reduce="max")
