"""Standard IrredNet backbone, with the original parameter names and initialization."""
import torch
from torch_scatter import scatter
from e3nn import o3
from .embedding import CrystalEmbeddingE3NN, GateTensorInteractionBlock
from .blocks import RealAgnosticTensorInteractionBlock
from .readout import FastTensorIrreps


class IrredNet(torch.nn.Module):
    """Standard IrredNet equivariant graph backbone. SO2 variants are outside this export."""

    def __init__(self, args):
        super().__init__()
        interaction_type = getattr(args, 'interaction_type', 'standard')
        if interaction_type != 'standard':
            raise ValueError('This standalone package supports interaction_type=standard only.')
        use_attention = getattr(args, 'use_attention', False)
        if use_attention:
            raise NotImplementedError('IrredNet.use_attention is disabled because the current attention block applies non-equivariant linear layers to irreps features.')
        self.num_layers = args.num_layers
        self.interaction_type = interaction_type
        self.use_attention = use_attention
        node_attrs_irreps = o3.Irreps(args.node_attrs_irreps)
        node_feats_irreps = o3.Irreps(args.node_feats_irreps)
        self.crystal_embedding = CrystalEmbeddingE3NN(node_attrs_irreps=node_attrs_irreps, node_feats_irreps=node_feats_irreps, max_radius=args.max_radius, num_of_basis=args.num_of_basis, num_polynomial_cutoff=args.num_polynomial_cutoff, radial_type=args.radial_type, distance_transform=args.distance_transform, lmax=args.lmax)
        sh_irreps = o3.Irreps.spherical_harmonics(args.lmax)
        hidden_irreps = o3.Irreps(args.hidden_irreps)
        self.gate_tensor_interaction = GateTensorInteractionBlock(node_attrs_irreps=node_attrs_irreps, node_feats_irreps=node_feats_irreps, num_layers=args.num_layers, sh_irreps=sh_irreps, edge_feats_irreps=self.crystal_embedding.edge_irreps, interaction_irreps=hidden_irreps, activation_type=args.activation_type, norm_type=args.norm_type, radial_MLP=args.radial_MLP)
        self.inter_last_layer = RealAgnosticTensorInteractionBlock(node_attrs_irreps=node_attrs_irreps, node_feats_irreps=self.gate_tensor_interaction.irreps_out, edge_attrs_irreps=sh_irreps, edge_feats_irreps=self.crystal_embedding.edge_irreps, target_irreps=hidden_irreps)
        self.TI = FastTensorIrreps(args.formula, hidden_irreps)

    def forward(self, data):
        node_attrs = data.x
        edge_vec = data.edge_vec
        node_feats, edge_feats, edge_attrs = self.crystal_embedding(node_attrs=node_attrs, edge_vec=edge_vec)
        interaction_kwargs = dict(node_attrs=node_attrs, node_feats=node_feats, edge_feats=edge_feats, edge_attrs=edge_attrs, edge_index=data.edge_index, batch=data.batch)
        node_feats = self.gate_tensor_interaction(**interaction_kwargs)
        last_kwargs = dict(node_attrs=node_attrs, node_feats=node_feats, edge_feats=edge_feats, edge_attrs=edge_attrs, edge_index=data.edge_index)
        node_feats = self.inter_last_layer(**last_kwargs)
        node_feats = scatter(node_feats, data.batch, dim=0, reduce='mean')
        node_feats = self.TI(node_feats, symmetry_projector=getattr(data, 'symmetry_projector', None), invariant_basis=getattr(data, 'invariant_basis', None), invariant_coeff_mask=getattr(data, 'invariant_coeff_mask', None))
        return node_feats
