import torch
import torch.nn as nn

from .bindformer import BindFormer
from .common import get_pos_CB, ATOM_CA
from .encoder import ProteinAtomEncoder, ProteinResidueEncoder


class Predictor(nn.Module):
    def __init__(self, feat_dim, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim)
        )

        self.project = nn.Linear(feat_dim, output_dim, bias=False)

    def forward(self, node_feat_wt, node_feat_mut, mask=None):
        """
        Args:
            node_feat_wt:   (N, L, F).
            node_feat_mut:  (N, L, F).
            mask:   (N, L).
        """
        feat_wm = torch.cat([node_feat_wt, node_feat_mut], dim=-1)
        feat_mw = torch.cat([node_feat_mut, node_feat_wt], dim=-1)
        feat_diff = self.mlp(feat_wm) - self.mlp(feat_mw)
        output_per_residue = self.project(feat_diff)
        if mask is not None:
            output_per_residue = output_per_residue * mask[...,None]
        output = output_per_residue.sum(dim=-2)
        return output


class UniBind(nn.Module):
    def __init__(self, output_dim, max_relpos=32, pair_feat_dim=64, node_feat_dim=12 * 14, geo_attn_num_layers=3, twotrack=False):
        super().__init__()
        self.twotrack = twotrack
        self.node_feat_dim = node_feat_dim
        self.atom_encoder = ProteinAtomEncoder(node_feat_dim)
        self.res_encoder = ProteinResidueEncoder(max_relpos=max_relpos, pair_feat_dim=pair_feat_dim, node_feat_dim=node_feat_dim)
        self.bindformer_blocks = nn.ModuleList()
        for _ in range(geo_attn_num_layers):
            bindformer_block = BindFormer(pair_feat_dim=pair_feat_dim, node_feat_dim=node_feat_dim, twotrack=twotrack)
            self.bindformer_blocks += [bindformer_block]
        self.predictor = Predictor(node_feat_dim, output_dim)
        self.num_classes = output_dim

    def forward(self, complex_wt, complex_mut):
        mask_atom_wt = complex_wt['pos14_mask']
        mask_atom_mut = complex_mut['pos14_mask']
        wt_energy = complex_wt.get('res_energy', None)
        mut_energy = complex_mut.get('res_energy', None)

        res_feat_wt, res_pair_feat_wt, res_pos14_wt, res_t_wt, res_R_wt = self.res_encoder(
            complex_wt['pos14'], complex_wt['aa'], 
            complex_wt['seq'], complex_wt['chain_seq'], wt_energy,
            mask_atom_wt, complex_wt['neighbors'])
        res_feat_mut, res_pair_feat_mut, res_pos14_mut, res_t_mut, res_R_mut = self.res_encoder(
            complex_mut['pos14'], complex_mut['aa'], 
            complex_mut['seq'], complex_mut['chain_seq'], mut_energy,
            mask_atom_mut, complex_mut['neighbors'])
        mask_residue_wt, mask_residue_mut = mask_atom_wt[:, :, ATOM_CA], mask_atom_mut[:, :, ATOM_CA]
        pos_cb_wt, pos_cb_mut = get_pos_CB(res_pos14_wt, mask_atom_wt), get_pos_CB(res_pos14_mut, mask_atom_mut)
        mask_res = mask_atom_wt[:, :, ATOM_CA]

        wt = {
            'complex': complex_wt,
            'res': {
                'feat': res_feat_wt,
                'pair_feat': res_pair_feat_wt,
                'R': res_R_wt,
                't': res_t_wt,
                'mask': mask_residue_wt,
                'pos_cb': pos_cb_wt,
            }
        }
        mut = {
            'complex': complex_mut,
            'res': {
                'feat': res_feat_mut,
                'pair_feat': res_pair_feat_mut,
                'R': res_R_mut,
                't': res_t_mut,
                'mask': mask_residue_mut,
                'pos_cb': pos_cb_mut
            }
        }
        if self.twotrack:
            atom_feat_wt, atom_R_wt, atom_t_wt = self.atom_encoder(complex_wt['aa'], complex_wt['pos14'], complex_wt['pos14_mask'])
            atom_feat_mut, atom_R_mut, atom_t_mut = self.atom_encoder(complex_mut['aa'], complex_mut['pos14'], complex_mut['pos14_mask'])
            wt['atom'] = {
                'feat': atom_feat_wt,
                'R': atom_R_wt,
                't': atom_t_wt
            }
            mut['atom'] = {
                'feat': atom_feat_mut,
                'R': atom_R_mut,
                't': atom_t_mut
            }
        for bindformer_block in self.bindformer_blocks:
            if self.twotrack:
                wt_res_feat, mut_res_feat, wt_atom_feat, mut_atom_feat = bindformer_block(wt, mut)
                wt['res']['feat'] = wt_res_feat
                mut['res']['feat'] = mut_res_feat
                wt['atom']['feat'] = wt_atom_feat
                mut['atom']['feat'] = mut_atom_feat
            else:
                wt_res_feat, mut_res_feat = bindformer_block(wt, mut)
                wt['res']['feat'] = wt_res_feat
                mut['res']['feat'] = mut_res_feat
        output = self.predictor(wt['res']['feat'], mut['res']['feat'], mask_res)
        return output


