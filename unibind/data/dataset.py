import dataclasses
import io

from torch import cdist
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.metrics import pairwise_distances

from unibind.data import proteins
from unibind.data.proteins import ProteinInput

ATOM_N, ATOM_CA, ATOM_C, ATOM_O, ATOM_CB = 0, 1, 2, 3, 4


def get_contact_dis_mat(pos_a, pos_b, mask_a, mask_b):
    '''
    contact distance matrix

    Args:
    ---------------------
    pos_a float[L1, n_atom, 3]
    pos_b float[L1, n_atom, 3]
    mask_a bool[L1, n_atom]
    mask_b bool[L1, n_atom]

    Returns:
    -----------------
    contact_dis_mat float[L1, L2]
    '''
    l1, n_atom = pos_a.shape[:2]
    l2, n_atom = pos_b.shape[:2]

    mask_a = mask_a.astype('bool')
    mask_b = mask_b.astype('bool')
    p1 = pos_a[mask_a] # [L_atom1, 3]
    p2 = pos_b[mask_b] # [L_atom2, 3]
    dis_mat = cdist(p1,p2) # [L_atom1, L_atom2]

    # [L, 37] -> [L_atom]
    index_a = np.tile(np.arange(l1)[:,None],reps=[1,n_atom])[mask_a]
    index_b = np.tile(np.arange(l2)[:,None],reps=[1,n_atom])[mask_b]

    # [L_atom1, L_atom2] -> [L1, L2]
    contact_dis_mat = dis_mat

    groups = index_a
    _ndx = np.argsort(groups)
    _id, _pos, g_count = np.unique(groups[_ndx], return_index=True, return_counts=True)
    contact_dis_mat = np.minimum.reduceat(contact_dis_mat[_ndx], _pos, axis=0) # [L_atom1, L_atom2] -> [L1, L_atom2]

    groups = index_b
    _ndx = np.argsort(groups)
    _id, _pos, g_count = np.unique(groups[_ndx], return_index=True, return_counts=True)
    contact_dis_mat = np.minimum.reduceat(contact_dis_mat[:,_ndx], _pos, axis=1) # [L1, L_atom2] -> [L1, L2]

    return contact_dis_mat


def single_mutation_parsing(mutation):
    aatype_from = mutation[0]
    chainid = mutation[1]
    position = int(mutation[2:-1])
    aatype_to = mutation[-1]
    return chainid, position, aatype_from, aatype_to


def mutation_parsing_to_frame(mutation):
    cols = 'chainid, position, aatype_from, aatype_to'.split(', ')
    mutations = []
    for m in mutation.split(','):
        mutations += [single_mutation_parsing(m)]
    df_mut = pd.DataFrame(mutations, columns=cols)
    df_mut['position'] = df_mut['position'].astype('int')
    return df_mut


def get_knn_dis_index(pos37, mask37, chainid, mutation_mask, chainids, n_neighbor, max_len=6000):
    pos37 = pos37[:max_len]
    mask37 = mask37[:max_len]
    chainid = chainid[:max_len]
    mutation_mask = mutation_mask[:max_len]

    pos_mut = pos37[mask37[:, 1] & mutation_mask]
    ca_wt = pos37[:, 1]
    ca_mut = pos_mut[:, 1]

    contact_dis_mat = pairwise_distances(ca_wt, ca_wt)
    if chainids is None:
        contact_dis_mat[chainid[None, :] == chainid[:, None]] = 10000
    else:
        isin = np.isin(chainid,chainids)
        contact_dis_mat[isin[None, :] == isin[:, None]] = 10000
    contact_dis_min = contact_dis_mat.min(axis=1)

    mut_dis_mat = pairwise_distances(ca_wt, ca_mut)
    mut_dis_min = mut_dis_mat.min(axis=1)

    dis_min = np.stack([contact_dis_min, mut_dis_min]).min(axis=0)
    dis_min_arg = dis_min.argsort()
    inter_res_index = dis_min_arg[:n_neighbor]
    return inter_res_index


def get_knn_dis_index_mat(pos14, n_neighbor=None):
    """
    pos14: [L, 14, 3] float
    inter_res_index: [L, n_neighbor] int
    """
    if n_neighbor == None:
        return np.arange(pos14.shape[0])
    ca_pos = pos14[:, ATOM_CA]

    dis_mat = pairwise_distances(ca_pos, ca_pos)
    dis_mat_zero_mask = dis_mat == 0
    dis_mat[dis_mat_zero_mask] = np.inf
    dis_min_arg = dis_mat.argsort(axis=-1)
    inter_res_index = dis_min_arg[:, :n_neighbor]
    return inter_res_index


class ComplexLoader(object):
    def __init__(self, data_root):
        super(ComplexLoader, self).__init__()
        self.data_root = Path(data_root)

    def get_protein(self, paths):
        pdb_strings = []
        for path in paths:
            path = self.data_root / path
            pdb_string = path.read_text()
            pdb_strings += [pdb_string]
        pdb_string = '\n'.join(pdb_strings)
        pdbs = proteins.ProteinInput.from_pdb(io.StringIO(pdb_string),
                                                   with_angles=False, return_dict=True)
        return pdbs

    def load(self, paths_wt, paths_mut, mutation, max_length, chainids='') -> (ProteinInput, ProteinInput):
        # load paths_wt, paths_mut
        if isinstance(paths_wt, str):
            paths_wt = paths_wt.split(',')
        if isinstance(paths_mut, str):
            paths_mut = paths_mut.split(',')
        pdbs_wt = self.get_protein(paths_wt)
        pdbs_mut = self.get_protein(paths_mut)
        chains = list(pdbs_wt.keys())

        for chain in chains:
            pdbs_wt[chain] = pdbs_wt[chain].mask_select(pdbs_wt[chain].mask == 1)
            pdbs_mut[chain] = pdbs_mut[chain].mask_select(pdbs_mut[chain].mask == 1)

        for chain in chains:
            mask = np.isin(pdbs_mut[chain].residue_index, pdbs_wt[chain].residue_index)
            pdbs_mut[chain] = pdbs_mut[chain].mask_select(mask)
            mask = np.isin(pdbs_wt[chain].residue_index, pdbs_mut[chain].residue_index)
            pdbs_wt[chain] = pdbs_wt[chain].mask_select(mask)

        df_mut = mutation_parsing_to_frame(mutation)
        df_mut['i'] = df_mut.apply(lambda x: pdbs_wt[x['chainid']].get_i_from_residue_index(x['position']), axis=1)
        mutation_masks = []
        for chain in chains:
            mutation_mask = np.zeros(pdbs_wt[chain].length)
            df_mut_sub = df_mut[df_mut['chainid'] == chain]
            mutation_mask[df_mut_sub['i']] = 1
            mutation_masks += [mutation_mask]

        p_wt: proteins.ProteinInput = proteins.proteins_merge([pdbs_wt[chain] for chain in chains],
                                                                        np.arange(len(chains)))
        p_mut: proteins.ProteinInput = proteins.proteins_merge([pdbs_mut[chain] for chain in chains],
                                                                         np.arange(len(chains)))
        if chainids == '':
            chainids = None
        else:
            chainids = chainids.split(',')
            chainids = [i for i, chain in enumerate(chains) if chain in chainids]
        mutation_mask = np.concatenate(mutation_masks).astype('bool')

        pos37 = p_wt.atom_positions
        mask37 = p_wt.atom_mask
        chainid = p_wt.chainid
        inter_res_index = get_knn_dis_index(pos37, mask37, chainid, mutation_mask, chainids, n_neighbor=max_length)
        mask = np.zeros(p_wt.length).astype('bool')
        mask[inter_res_index] = True

        p14_wt = p_wt.to_atom14()
        p14_wt = p14_wt.mask_select(mask)
        p14_wt = p14_wt.padding(max_length - p14_wt.length)

        p14_mut = p_mut.to_atom14()
        p14_mut = p14_mut.mask_select(mask)
        p14_mut = p14_mut.padding(max_length - p14_mut.length)

        center = p14_wt.get_center()
        p14_wt = p14_wt.translation(-center)
        p14_mut = p14_mut.translation(-center)

        return p14_wt, p14_mut, chains


class UnibindDataset(Dataset):
    def __init__(self, df_path, data_root, max_length=256,
                 col_wt='path_wt', col_mut='path_mut', col_mutation='mutation', col_chainids='chainids',
                 cols_label=None,
                 n_neighbors=None,
                 train=True, diskcache=None, ):
        if cols_label is None:
            cols_label = ['ddg']
        self.data_loader = ComplexLoader(data_root)

        if isinstance(df_path, pd.DataFrame):
            self.df = df_path.copy()
        else:
            self.df = pd.read_csv(df_path, low_memory=False)
        if col_chainids not in self.df:
            self.df[col_chainids] = ''
        self.df[col_chainids] = self.df[col_chainids].fillna('').astype('str')

        self.max_length = max_length
        self.n_neighbors = n_neighbors

        self.col_wt = col_wt
        self.col_mut = col_mut
        self.col_mutation = col_mutation
        self.col_chainids = col_chainids
        self.cols_label = cols_label
        self.train = train
        self.diskcache = diskcache

        for col in self.cols_label:
            self.df[col] = self.df.get(col, np.nan)
        self.num_classes = len(self.cols_label)

    def __len__(self):
        return len(self.df)

    def load_data(self, paths_wt, paths_mut, mutation, max_length, chainids, cache):
        key = paths_wt, paths_mut, mutation, max_length, chainids

        if cache is None:
            values = self.data_loader.load(*key)
            return values

        if key in cache:
            values = cache[key]
            return values

        values = self.data_loader.load(*key)
        cache[key] = values
        return values

    def get_data(self, idx):
        case = self.df.iloc[idx]
        paths_wt = case[self.col_wt]
        paths_mut = case[self.col_mut]
        mutation = case[self.col_mutation]
        chainids = case[self.col_chainids]
        cache = self.diskcache

        paths_wt = ','.join(sorted(paths_wt.split(',')))
        paths_mut = ','.join(sorted(paths_mut.split(',')))

        values = self.load_data(paths_wt, paths_mut, mutation, self.max_length, chainids, cache)
        p14_wt, p14_mut, chains = values

        labels = case[self.cols_label].fillna(0).astype('float32').values
        labels_valid_mask = case[self.cols_label].notna().values

        if isinstance(chainids, str) and (chainids != ''):
            indices = [chains.index(chainid) if chainid in chains else -1 for chainid in chainids]
            indices = [i for i in indices if i != -1]
            chainid_wt_new = 1 - np.isin(p14_wt.chainid, indices).astype('int')
            p14_wt = dataclasses.replace(p14_wt, chainid=chainid_wt_new)
            p14_mut = dataclasses.replace(p14_mut, chainid=chainid_wt_new)

        complex_wt = {
            'pos14': p14_wt.atom_positions.astype('float32'),
            'pos14_mask': p14_wt.atom_mask.astype('bool'),
            'aa': p14_wt.aatype.astype('int'),
            'seq': p14_wt.residue_index.astype('int'),
            'chain_seq': p14_wt.chainid.astype('int'),
        }
        complex_mut = {
            'pos14': p14_mut.atom_positions.astype('float32'),
            'pos14_mask': p14_mut.atom_mask.astype('bool'),
            'aa': p14_mut.aatype.astype('int'),
            'seq': p14_mut.residue_index.astype('int'),
            'chain_seq': p14_mut.chainid.astype('int'),
        }
        complex_wt['neighbors'] = get_knn_dis_index_mat(complex_wt['pos14'], self.n_neighbors)
        complex_mut['neighbors'] = get_knn_dis_index_mat(complex_mut['pos14'], self.n_neighbors)

        label = (labels.astype('float32'), labels_valid_mask.astype('bool'))

        return (complex_wt, complex_mut), label

    def __getitem__(self, idx):
        try:
            return self.get_data(idx)
        except Exception as e:
            case = self.df.iloc[idx]
            paths_wt = case[self.col_wt].split(',')
            paths_mut = case[self.col_mut].split(',')
            ddg = case[self.col_label]
            print(case, paths_wt, paths_mut, ddg)
            raise e
