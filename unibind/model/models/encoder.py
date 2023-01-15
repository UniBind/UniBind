import torch
import torch.nn as nn

from .common import construct_3d_basis, global_to_local, ATOM_CA
from .common import ATOM_C, ATOM_N


def get_neighbors(a, neighbors):
    """
    a: B x L
    neighbors: B x L x M
    result: B x L x M
    """
    B, L = a.shape
    a = a[:, :, None].expand(-1, -1, L)
    result = a.gather(1, neighbors)
    return result

def get_neighbors_z(a, neighbors):
    """
    a: B x L x L x D
    neighbors: B x L x M, M: No. of neighbors
    result: B x L x M x D
    """
    B, L, LL, D = a.shape
    neighbors = neighbors[:, :, :, None].expand(-1, -1, -1, D)
    result = a.gather(2, neighbors)
    return result


class PositionalEncoding(nn.Module):

    def __init__(self, num_funcs=6):
        super().__init__()
        self.num_funcs = num_funcs
        self.register_buffer('freq_bands', 2.0 ** torch.linspace(0.0, num_funcs-1, num_funcs))

    def get_out_dim(self, in_dim):
        return in_dim * (2 * self.num_funcs + 1)

    def forward(self, x):
        """
        Args:
            x:  (..., d).
        """
        shape = list(x.shape[:-1]) + [-1]
        x = x.unsqueeze(-1) # (..., d, 1)
        code = torch.cat([x, torch.sin(x * self.freq_bands), torch.cos(x * self.freq_bands)], dim=-1)   # (..., d, 2f+1)
        code = code.reshape(shape)
        return code

class ResidueEmbedding(nn.Module):

    def __init__(self, feat_dim):
        super().__init__()
        self.aatype_embed = nn.Embedding(22, feat_dim) # 2022年04月03日 多一个未知，padding
        self.torsion_embed = PositionalEncoding()
        self.mlp = nn.Sequential(
            nn.Linear(21*14*3 + feat_dim, feat_dim * 2), nn.ReLU(),
            nn.Linear(feat_dim * 2, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim)
        )

    def forward(self, aa, pos14, atom_mask):
        """
        Args:
            aa:           (N, L).
            pos14:        (N, L, 14, 3).
            atom_mask:    (N, L, 14).
        """
        N, L = aa.size()
        device = aa.device

        R = construct_3d_basis(pos14[:, :, 1], pos14[:, :, 2], pos14[:, :, 0])  # (N, L, 3, 3)
        t = pos14[:, :, 1]  # (N, L, 3)
        crd14 = global_to_local(R, t, pos14)    # (N, L, 14, 3)
        crd14_mask = atom_mask[:, :, :, None].expand_as(crd14)
        crd14 = torch.where(crd14_mask, crd14, torch.zeros_like(crd14))

        aa_expand  = aa[:, :, None, None, None].expand(N, L, 21, 14, 3)
        rng_expand = torch.arange(0, 21, device=device)[None, None, :, None, None].expand(N, L, 21, 14, 3).to(aa_expand)
        place_mask = (aa_expand == rng_expand)
        crd_expand = crd14[:, :, None, :, :].expand(N, L, 21, 14, 3)
        crd_expand = torch.where(place_mask, crd_expand, torch.zeros_like(crd_expand))
        crd_feat = crd_expand.reshape(N, L, 21 * 14 * 3)

        aa_feat = self.aatype_embed(aa) # (N, L, feat)

        out_feat = self.mlp(torch.cat([crd_feat, aa_feat], dim=-1))
        return out_feat


class ProteinAtomEncoder(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim // 14 * 2 + 3, feat_dim // 14 * 2), nn.ReLU(),
            nn.Linear(feat_dim // 14 * 2, feat_dim // 14)
        )
        self.atom_embed = nn.Embedding(22, feat_dim)
        self.atom_type_embed = nn.Parameter(torch.empty((14, feat_dim // 14)))
        nn.init.normal_(self.atom_type_embed)

    def forward(self, aa, pos14, pos14_mask):
        """
        Args:
            aa:           (N, L).
            pos14:        (N, L, 14, 3).
        """
        N, L = aa.size()
        atom_pos = pos14.view(N, L * 14, 3)
        atom_embed = self.atom_embed(aa).reshape(N, L, 14, -1)
        atom_type_embed_expanded = self.atom_type_embed[None, None, :, :].expand(N, L, -1, -1)
        out_feat = self.mlp(torch.cat([atom_type_embed_expanded, 
                                       atom_embed,
                                       pos14], dim=-1)) 
        out_feat[~pos14_mask[..., None].expand_as(out_feat)] = 0

        c_pos = pos14[:, :, ATOM_C, :]
        c_pos = c_pos[:, :, None, :].expand(-1, -1, 14, -1).reshape(N, L * 14, 3)

        n_pos = pos14[:, :, ATOM_N, :]
        n_pos = n_pos[:, :, None, :].expand(-1, -1, 14, -1).reshape(N, L * 14, 3)

        R = construct_3d_basis(atom_pos, c_pos, n_pos)
        t = atom_pos
        return out_feat, R.view(N, L, 14, 3, 3), t.view(N, L, 14, 3)


class ProteinResidueEncoder(nn.Module):

    def __init__(self, max_relpos=32, pair_feat_dim=64, node_feat_dim=64, geo_attn_num_layers=2):
        super().__init__()

        self.max_relpos = max_relpos
        self.relpos_embedding = nn.Embedding(max_relpos*2+2, pair_feat_dim)
        self.energy_proj = nn.Linear(7, pair_feat_dim, bias=False)
        self.residue_encoder = ResidueEmbedding(node_feat_dim)

    def forward(self, pos14, aa, seq, chain, energy, mask_atom, neighbors):
        """
        Args:
            pos14:  (N, L, 14, 3).
            aa:     (N, L).
            seq:    (N, L).
            chain:  (N, L).
            energy: (N, L, L, 7)
            mask_atom:  (N, L, 14)
        Returns:
            (N, L, node_ch)
        """
        chain_knn = get_neighbors(chain, neighbors)
        same_chain = (chain[:, :, None] == chain_knn)   # (N, L, M)
        relpos = (seq[:, :, None] - chain_knn).clamp(min=-self.max_relpos, max=self.max_relpos) + self.max_relpos # (N, L, M)
        relpos = torch.where(same_chain, relpos, torch.full_like(relpos, fill_value=self.max_relpos*2+1))
        pair_feat = self.relpos_embedding(relpos)   # (N, L, M, pair_ch)
        if energy is not None:
            pair_feat = pair_feat + self.energy_proj(get_neighbors_z(energy, neighbors))
        R = construct_3d_basis(pos14[:, :, ATOM_CA], pos14[:, :, ATOM_C], pos14[:, :, ATOM_N])
        # pair_feat = None
        # R = None

        # Residue encoder
        res_feat = self.residue_encoder(aa, pos14, mask_atom)

        # # Geom encoder
        t = pos14[:, :, ATOM_CA]

        return res_feat, pair_feat, pos14, t, R

