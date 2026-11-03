"""Crystal embedding and gated equivariant interactions used by IrredNet."""

import ast
import torch
from e3nn import o3
from .blocks import (LinearNodeEmbeddingBlock, RadialEmbeddingBlock,
                     RealAgnosticTensorInteractionBlock, GateLayer, NormalizationLayer)


class GateTensorInteractionBlock(torch.nn.Module):
    def __init__(self, 
                 node_attrs_irreps,
                 node_feats_irreps,
                 num_layers,
                 sh_irreps, 
                 edge_feats_irreps,
                 interaction_irreps,
                 activation_type,
                 norm_type, 
                 radial_MLP,
                 ):
        super().__init__()

        self.inter_layer_list = torch.nn.ModuleList()
        self.gate_layer_list = torch.nn.ModuleList()
        self.norm_layer_list = torch.nn.ModuleList()
        self.num_layers = num_layers
        for i in range(num_layers - 1):
            if i == 0:
                node_gate_irreps_in = node_feats_irreps
            else:
                node_feats_irreps = self.gate_layer_list[i-1].irreps_out
                node_gate_irreps_in = self.inter_layer_list[i-1].irreps_out
            
            # gate layer
            gate = GateLayer(
                tp_irreps_in1 = node_gate_irreps_in, 
                tp_irreps_in2 = sh_irreps, 
                tp_irreps_out = interaction_irreps, 
                activation_type = activation_type,
            )
            self.gate_layer_list.append(gate)   
            inter_layer = RealAgnosticTensorInteractionBlock(
                node_attrs_irreps = node_attrs_irreps,
                node_feats_irreps = node_feats_irreps,
                edge_attrs_irreps = sh_irreps,
                edge_feats_irreps = edge_feats_irreps,
                target_irreps = gate.irreps_in,
                radial_MLP = ast.literal_eval(radial_MLP),
            )
            self.inter_layer_list.append(inter_layer)
            norm_layer = NormalizationLayer(
                irreps = gate.irreps_out,
                norm_type = norm_type,
            )
            self.norm_layer_list.append(norm_layer)
    
    def forward(self, node_attrs, node_feats, edge_feats, edge_attrs, edge_index, batch):
        for i in range(self.num_layers - 1):
            node_feats = self.inter_layer_list[i](
                node_attrs = node_attrs,
                node_feats = node_feats,
                edge_feats = edge_feats,
                edge_attrs = edge_attrs,
                edge_index = edge_index,
            )
            node_feats = self.gate_layer_list[i](
                node_feats = node_feats,
            )
            node_feats = self.norm_layer_list[i](
                node_feats = node_feats,
                batch = batch,
            )
        return node_feats
    
    @property
    def irreps_out(self):
        return self.gate_layer_list[-1].irreps_out


class CrystalEmbeddingE3NN(torch.nn.Module):
    """
    Crystal embedding module using E3NN modules.
    """
    def __init__(self, 
                 node_attrs_irreps,
                 node_feats_irreps,
                 max_radius,
                 num_of_basis,
                 num_polynomial_cutoff,
                 radial_type,
                 distance_transform,
                 lmax,
                 ):
        super().__init__()
        # node embedding parameters
        node_attrs_irreps = o3.Irreps(node_attrs_irreps)
        node_feats_irreps = o3.Irreps(node_feats_irreps)
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in = node_attrs_irreps,
            irreps_out = node_feats_irreps
        )
        # edge distance embedding paramters
        self.radial_embedding = RadialEmbeddingBlock(
            r_max = max_radius,
            num_bessel = num_of_basis,
            num_polynomial_cutoff= num_polynomial_cutoff,
            radial_type= radial_type,
            distance_transform= distance_transform,
        )
        # edge angular embedding paramters
        sh_irreps = o3.Irreps.spherical_harmonics(lmax)
        self.spherical_harmonics = o3.SphericalHarmonics(
            irreps_out = sh_irreps, 
            normalize=True, 
            normalization='component',
        )

    def forward(self, 
                node_attrs,
                edge_vec,
                ):
        # node embedding
        node_feats = self.node_embedding(node_attrs)
        # edge distance embedding
        edge_lengths = edge_vec.norm(dim=1).unsqueeze(1)
        edge_feats = self.radial_embedding(edge_lengths)
        edge_attrs = self.spherical_harmonics(edge_vec)
        return node_feats, edge_feats, edge_attrs
    
    @property
    def edge_irreps(self):
        return o3.Irreps(f"{self.radial_embedding.out_dim}x0e")
