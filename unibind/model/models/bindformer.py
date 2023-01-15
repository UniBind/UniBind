import torch
import torch.nn as nn
from .gea import GEAEncoderResidue, GEAEncoderAtom


class BindFormer(nn.Module):
    def __init__(self, pair_feat_dim=64, node_feat_dim=12 * 14, twotrack=False):
        super().__init__()
        self.twotrack = twotrack
        self.node_feat_dim = node_feat_dim
        if self.twotrack:
            self.res2atoms = nn.Sequential(
                nn.Linear(node_feat_dim, node_feat_dim), nn.ReLU(),
                nn.Linear(node_feat_dim, node_feat_dim)
            )
            self.atoms2res = nn.Sequential(
                nn.Linear(node_feat_dim, node_feat_dim), nn.ReLU(),
                nn.Linear(node_feat_dim, node_feat_dim)
            )
            self.atom_gea_layer = GEAEncoderAtom(
                node_feat_dim = node_feat_dim,
                pair_feat_dim = pair_feat_dim,
                num_layers = 1
            )
            self.atom_mlp = nn.Sequential(
                nn.Linear(node_feat_dim // 14, node_feat_dim // 14), nn.ReLU(),
                nn.Linear(node_feat_dim // 14, node_feat_dim // 14)
            )
            self.atom_norm = nn.ModuleList([nn.LayerNorm(node_feat_dim // 14) for _ in range(3)])

        self.res_gea_layer = GEAEncoderResidue(
            node_feat_dim = node_feat_dim,
            pair_feat_dim = pair_feat_dim,
            num_layers = 1
        )
        self.res_mlp = nn.Sequential(
            nn.Linear(node_feat_dim, node_feat_dim), nn.ReLU(),
            nn.Linear(node_feat_dim, node_feat_dim)
        )
        self.res_norm = nn.ModuleList([nn.LayerNorm(node_feat_dim) for _ in range(3)])

    def _update_twotrack(self, wt, mut):
        res_ga_layer = self.res_gea_layer
        res_mlp = self.res_mlp
        res_norm = self.res_norm
        atom_ga_layer = self.atom_gea_layer
        atom_mlp = self.atom_mlp
        atom_norm = self.atom_norm

        N, L = wt['complex']['aa'].shape[:2]

        wt_res_feat = wt['res']['feat']
        mut_res_feat = mut['res']['feat']
        wt_atom_feat = wt['atom']['feat']
        mut_atom_feat = mut['atom']['feat']

        wt_atom_feat = wt_atom_feat + self.res2atoms(wt_res_feat-mut_res_feat).view(N, L, 14, -1)
        mut_atom_feat = mut_atom_feat + self.res2atoms(mut_res_feat-wt_res_feat).view(N, L, 14, -1)
        wt_atom_feat = atom_norm[0](wt_atom_feat)
        mut_atom_feat = atom_norm[0](mut_atom_feat)

        wt_atom_feat = wt_atom_feat + atom_ga_layer(wt['atom']['R'], wt['atom']['t'], wt['complex']['pos14'],
                                                    wt_atom_feat, wt['res']['pair_feat'],
                                                    wt['complex']['pos14_mask'], wt['complex']['neighbors'])
        mut_atom_feat = mut_atom_feat + atom_ga_layer(mut['atom']['R'], mut['atom']['t'], mut['complex']['pos14'],
                                                      mut_atom_feat, mut['res']['pair_feat'],
                                                      mut['complex']['pos14_mask'], mut['complex']['neighbors'])
        wt_atom_feat = atom_norm[1](wt_atom_feat)
        mut_atom_feat = atom_norm[1](mut_atom_feat)

        wt_atom_feat = wt_atom_feat + atom_mlp(wt_atom_feat-mut_atom_feat)
        mut_atom_feat = mut_atom_feat + atom_mlp(mut_atom_feat-wt_atom_feat)
        wt_atom_feat = atom_norm[2](wt_atom_feat)
        mut_atom_feat = atom_norm[2](mut_atom_feat)

        wt_res_feat = wt_res_feat + self.atoms2res(wt_atom_feat.view(N, L, -1) - mut_atom_feat.view(N, L, -1))
        mut_res_feat = mut_res_feat + self.atoms2res(mut_atom_feat.view(N, L, -1) - wt_atom_feat.view(N, L, -1))
        wt_res_feat = res_norm[0](wt_res_feat)
        mut_res_feat = res_norm[0](mut_res_feat)

        wt_res_feat = wt_res_feat + res_ga_layer(wt['res']['R'], wt['res']['t'], wt['res']['pos_cb'],
                                                 wt_res_feat, wt['res']['pair_feat'],
                                                 wt['res']['mask'], wt['complex']['neighbors'])
        mut_res_feat = mut_res_feat + res_ga_layer(mut['res']['R'], mut['res']['t'], mut['res']['pos_cb'],
                                                   mut_res_feat, mut['res']['pair_feat'],
                                                   mut['res']['mask'], mut['complex']['neighbors'])
        wt_res_feat = res_norm[1](wt_res_feat)
        mut_res_feat = res_norm[1](mut_res_feat)

        wt_res_feat = wt_res_feat + res_mlp(wt_res_feat-mut_res_feat)
        mut_res_feat = mut_res_feat + res_mlp(mut_res_feat-wt_res_feat)
        wt_res_feat = res_norm[2](wt_res_feat)
        mut_res_feat = res_norm[2](mut_res_feat)
        return wt_res_feat, mut_res_feat, wt_atom_feat, mut_atom_feat

    def _update(self, wt, mut):
        res_feat_wt = wt['res']['feat']
        res_feat_mut = mut['res']['feat']
        res_feat_wt = res_feat_wt + self.res_mlp(res_feat_wt-res_feat_mut)
        res_feat_mut = res_feat_mut + self.res_mlp(res_feat_mut-res_feat_wt)
        res_feat_wt = self.res_ga_layer(wt['res']['R'], wt['res']['t'], wt['res']['pos_cb'],
                                        res_feat_wt, wt['res']['pair_feat'],
                                        wt['res']['mask'], wt['complex']['neighbors'])
        res_feat_mut = self.res_ga_layer(mut['res']['R'], mut['res']['t'], mut['res']['pos_cb'],
                                         res_feat_mut, mut['res']['pair_feat'],
                                         mut['res']['mask'], mut['complex']['neighbors'])
        res_feat_wt = self.res_norm(res_feat_wt)
        res_feat_mut = self.res_norm(res_feat_mut)
        return res_feat_wt, res_feat_mut

    def forward(self, wt, mut):
        if self.twotrack:
            wt_res_feat, mut_res_feat, wt_atom_feat, mut_atom_feat = self._update_twotrack(wt, mut)
            return wt_res_feat, mut_res_feat, wt_atom_feat, mut_atom_feat
        else:
            wt_res_feat, mut_res_feat = self._update(wt, mut)
            return wt_res_feat, mut_res_feat


