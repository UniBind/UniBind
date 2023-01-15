from pathlib import Path
import dataclasses
from dataclasses import dataclass, field
from typing import Optional
import io

import numpy as np
import scipy
from Bio.PDB import PDBParser

from unibind.utils.residue_constants import restypes_with_x, restype_order_with_x, STANDARD_ATOM_MASK, restype_3to1, \
    restype_order, restype_num, atom_type_num, atom_types, atom_order
from unibind.utils.atom_convert import atom37_to_torsion7_np, atom37_to_atom14


@dataclass
class ProteinInput:
    seq: str  # L
    mask: np.ndarray  # (L, )
    aatype: np.ndarray  # (L, )
    atom_mask: np.ndarray  # (L, 37)

    atom_positions: Optional[np.ndarray] = field(default=None)  # (L, 37, 3)
    residue_index: Optional[np.ndarray] = field(default=None)  # (L, )
    b_factors: Optional[np.ndarray] = field(default=None)  # (L, 37)

    torsion_angles_sin_cos: Optional[np.ndarray] = field(default=None)  # (L, 7, 2)
    alt_torsion_angles_sin_cos: Optional[np.ndarray] = field(default=None)  # (L, 7, 2)
    torsion_angles_mask: Optional[np.ndarray] = field(default=None)  # (L, 7)

    chainid: Optional[np.ndarray] = field(default=None)  # (L)

    def __post_init__(self):
        assert not all([a is None for a in [self.seq, self.aatype]])
        if (self.seq is None) and (self.aatype is not None):
            self.seq = ''.join([restypes_with_x[a] for a in self.aatype.clip(0, 20)])
        if (self.seq is not None) and (self.aatype is None):
            self.aatype = np.array([restype_order_with_x.get(a, 21) for a in self.seq], dtype='int64')

        if self.atom_mask is None:
            self.atom_mask = STANDARD_ATOM_MASK[self.aatype].astype('bool')
        if self.mask is None:
            self.mask = self.atom_mask[:, 1].astype('bool')

        self.aatype = self.aatype.astype('int64')
        self.mask = self.mask.astype('bool')
        self.atom_mask = self.atom_mask.astype('bool')

        if self.b_factors is not None:
            if len(self.b_factors.shape) == 1:
                self.b_factors = self.b_factors[:, None] * self.atom_mask
            assert len(self.b_factors.shape) == 2

        assert len(self.mask.shape) == 1
        assert len(self.atom_mask.shape) == 2

    @property
    def length(self):
        return len(self.seq)

    @property
    def length_valid(self):
        return self.mask.sum().item()

    def has_structure(self):
        return (self.atom_positions is not None) and (self.mask.sum() != 0)

    def to_atom14(self):
        if self.atom_positions.shape[1] == 14:
            return self
        else:
            atom_positions, mask14, arrs = atom37_to_atom14(self.aatype, self.atom_positions,
                                                            [self.atom_mask, self.b_factors])
            atom_mask, b_factors = arrs
            atom_mask = atom_mask * mask14
            b_factors = b_factors * mask14
            return dataclasses.replace(self, atom_positions=atom_positions, atom_mask=atom_mask, b_factors=b_factors)

    def fillna(self, with_angles=True):
        length = len(self.seq)
        result = {}
        if self.atom_positions is None:
            result['atom_positions'] = np.zeros([length, 37, 3], dtype='float32')
        if self.residue_index is None:
            result['residue_index'] = np.arange(length).astype('int64')
        if self.b_factors is None:
            result['b_factors'] = np.zeros([length, 37], dtype='float32')
        if with_angles:
            if self.atom_positions is None:
                result_t = dict(
                    torsion_angles_sin_cos=np.zeros([length, 7, 2], dtype='float32'),
                    alt_torsion_angles_sin_cos=np.zeros([length, 7, 2], dtype='float32'),
                    torsion_angles_mask=np.zeros([length, 7], dtype='bool'),
                )
            elif self.torsion_angles_sin_cos is None:
                sin_cos, sin_cos_mask, alt_sin_cos = atom37_to_torsion7_np(self.aatype, self.atom_positions,
                                                                           self.atom_mask)
                result_t = {
                    'torsion_angles_sin_cos': sin_cos,
                    'alt_torsion_angles_sin_cos': sin_cos_mask,
                    'torsion_angles_mask': alt_sin_cos,
                }
            else:
                result_t = {}
            result.update(result_t)
        return dataclasses.replace(self, **result)

    def padding(self, pad_width):
        if pad_width > 0:
            values_base = dict(
                seq=self.seq + 'X' * pad_width,
                mask=np.pad(self.mask, ((0, pad_width)), mode='constant', constant_values=False),
                aatype=np.pad(self.aatype, ((0, pad_width)), mode='constant', constant_values=21),
                atom_mask=np.pad(self.atom_mask, ((0, pad_width), (0, 0)), mode='constant', constant_values=False),
            )
            array_padding_settings = {
                'atom_positions': dict(pad_width=((0, pad_width), (0, 0), (0, 0)), mode='edge'),
                'residue_index': dict(pad_width=((0, pad_width)), mode='edge'),
                'b_factors': dict(pad_width=((0, pad_width), (0, 0)), mode='edge'),
                'torsion_angles_sin_cos': dict(pad_width=((0, pad_width), (0, 0), (0, 0)), mode='edge'),
                'alt_torsion_angles_sin_cos': dict(pad_width=((0, pad_width), (0, 0), (0, 0)), mode='edge'),
                'torsion_angles_mask': dict(pad_width=((0, pad_width), (0, 0)), mode='edge'),
                'chainid': dict(pad_width=((0, pad_width)), mode='constant', constant_values=20),
            }
            values = {}
            for k in array_padding_settings.keys():
                value = getattr(self, k)
                if value is not None:
                    value = np.pad(value, **array_padding_settings[k])
                    values[k] = value
            values_all = values_base
            values_all.update(values)
            return dataclasses.replace(self, **values_all)
        else:
            return self

    def slice(self, begin, end):
        if (begin < 0):
            raise Exception(f'error begin: {begin}')
        if end > self.length:
            pad_width = end - self.length
            self = self.padding(pad_width)
        names = set(self.__dataclass_fields__.keys())
        result = {}
        for name in names:
            value = getattr(self, name)
            if value is None:
                result[name] = value
            else:
                result[name] = value[begin:end]
        return dataclasses.replace(self, **result)

    def mask_select(self, mask):
        mask = mask.astype('bool')
        assert len(mask) == self.length

        names = list(self.__dataclass_fields__.keys())
        result = {}
        for name in names:
            value = getattr(self, name)
            if name == 'seq':
                result[name] = ''.join([r for r, m in zip(value, mask) if m == 1])
            elif value is None:
                result[name] = value
            else:
                result[name] = value[mask]
        return dataclasses.replace(self, **result)

    def __getitem__(self, key):
        if isinstance(key, slice):
            begin = key.start if key.start else 0
            end = key.stop if key.stop else self.length
            if (begin < 0) or (end > self.length):
                raise Exception(f'error span: {key}')
            return self.slice(begin, end)
        else:
            raise TypeError('Index must be slice, not {}'.format(type(key).__name__))

    def append(self, other):
        result = dict(
            seq=self.seq + other.seq,
            mask=np.concatenate([self.mask, other.mask]),
            aatype=np.concatenate([self.aatype, other.aatype]),
            atom_mask=np.concatenate([self.atom_mask, other.atom_mask]),
        )
        names = set(self.__dataclass_fields__.keys())
        for name in names:
            if name in result:
                continue
            value = getattr(self, name)
            if value is None:
                result[name] = value
            else:
                result[name] = np.concatenate([value, getattr(other, name)])
        return dataclasses.replace(self, **result)

    def append_list(self, others):
        if len(others) == 0:
            return self
        result = dict(
            seq=''.join([self.seq] + [other.seq for other in others]),
            mask=np.concatenate([self.mask] + [other.mask for other in others]),
            aatype=np.concatenate([self.aatype] + [other.aatype for other in others]),
            atom_mask=np.concatenate([self.atom_mask] + [other.atom_mask for other in others]),
        )
        names = set(self.__dataclass_fields__.keys())
        for name in names:
            if name in result:
                continue
            value = getattr(self, name)
            if value is None:
                result[name] = value
            else:
                result[name] = np.concatenate([value] + [getattr(other, name) for other in others])
        return dataclasses.replace(self, **result)

    @classmethod
    def from_pdbchain(cls, chain):
        result = chain2arrays(chain)
        return cls(**result)

    @classmethod
    def from_pdb(cls, path, with_angles=True, return_dict=False, verbose=False):
        if isinstance(path, io.IOBase):
            file_string = path.read()
        else:
            path = Path(path)
            file_string = path.read_text()

        proteins = {}
        chains = chains_from_pdb_string(file_string)
        for chain in chains:
            try:
                chainid = chain.id
                protein = cls.from_pdbchain(chain)
                if with_angles:
                    protein = protein.fillna(with_angles=True)
                proteins[chainid] = protein
            except Exception as e:
                if verbose:
                    print(path)
                    print(chain)
                    print(e)

        if not return_dict:
            protein = next(iter(proteins.values()))
            return protein
        else:
            return proteins

    def to_dict(self):
        return dataclasses.asdict(self)

    def get_center(self):
        atom_positions = self.atom_positions
        atom_mask = self.atom_mask
        if atom_mask is None:
            center = np.zeros(3)
        elif atom_mask.sum() == 0:
            center = np.zeros(3)
        else:
            center = atom_positions[atom_mask].mean(axis=0)
        return center

    def translation(self, translation):
        atom_positions = self.atom_positions
        atom_positions = atom_positions + translation[None, None, :]
        return dataclasses.replace(self, atom_positions=atom_positions)

    def rotation(self, rotation, around_point=None):
        atom_positions = self.atom_positions
        r = scipy.spatial.transform.Rotation.from_matrix(rotation)
        if around_point is not None:
            atom_positions_0 = atom_positions - around_point[None, None, :]
            atom_positions_new = r.apply(atom_positions_0.reshape(-1, 3)).reshape(-1, 37, 3)
            atom_positions_new = atom_positions_new + around_point[None, None, :]
        else:
            atom_positions_new = r.apply(atom_positions.reshape(-1, 3)).reshape(-1, 37, 3)

        return dataclasses.replace(self, atom_positions=atom_positions_new)

    def get_i_from_residue_index(self, index, around=False):
        i = np.searchsorted(self.residue_index, index)
        if i >= self.length:
            return -1
        elif index != self.residue_index[i] and (not around):
            return -1
        else:
            return i

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        texts = []
        texts += [f'seq: {self.seq}']
        texts += [f'length: {len(self.seq)}']
        texts += [f"mask: {''.join(self.mask.astype('int').astype('str'))}"]
        if self.chainid is not None:
            texts += [f"chainid: {''.join(self.chainid.astype('int').astype('str'))}"]

        names = [
            'aatype',
            'atom_mask',
            'atom_positions',
            'residue_index',
            'b_factors',
            'torsion_angles_sin_cos',
            'alt_torsion_angles_sin_cos',
            'torsion_angles_mask',
        ]
        for name in names:
            value = getattr(self, name)
            if value is None:
                text = f'{name}: None'
            else:
                text = f'{name}: {value.shape}'
            texts += [text]
        text = ', \n  '.join(texts)
        text = f'Protein(\n  {text}\n)'
        return text


def proteins_merge(proteins, chainids=None):
    assert len(proteins) > 0
    p = proteins[0]
    p = p.append_list(proteins[1:])
    if chainids is None:
        chain_arr = np.concatenate([[i] * p.length for i, p in enumerate(proteins)]).astype('int')
    else:
        chain_arr = np.concatenate([[i] * p.length for i, p in zip(chainids, proteins)]).astype('int')
    p = dataclasses.replace(p, chainid=chain_arr)
    return p


def seq2aatype(seq):
    aatype = np.array([restype_order_with_x.get(a, 21) for a in seq], dtype='int64')
    return aatype


def chains_from_pdb_string(pdb_str: str):
    pdb_fh = io.StringIO(pdb_str)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("none", pdb_fh)
    models = list(structure.get_models())
    if len(models) != 1:
        raise ValueError(
            f"Only single model PDBs are supported. Found {len(models)} models."
        )
    model = models[0]
    chains = list(model.get_chains())
    return chains


def chain2arrays(chain):
    seq = []
    atom_positions = []
    aatype = []
    atom_mask = []
    residue_index = []
    b_factors = []
    for res in chain:
        if res.id[0] != ' ':
            continue
        res_shortname = restype_3to1.get(res.resname, 'X')
        restype_idx = restype_order.get(
            res_shortname, restype_num)
        pos = np.zeros((atom_type_num, 3))
        mask = np.zeros((atom_type_num,))
        res_b_factors = np.zeros((atom_type_num,))
        for atom in res:
            if atom.name not in atom_types:
                continue
            pos[atom_order[atom.name]] = atom.coord
            mask[atom_order[atom.name]] = 1.
            res_b_factors[atom_order[atom.name]] = atom.bfactor
        if np.sum(mask) < 0.5:
            continue
        seq.append(res_shortname)
        aatype.append(restype_idx)
        atom_positions.append(pos)
        atom_mask.append(mask)
        residue_index.append(res.id[1])
        b_factors.append(res_b_factors)
    seq = ''.join(seq)
    atom_positions = np.array(atom_positions)
    atom_mask = np.array(atom_mask)
    aatype = np.array(aatype)
    residue_index = np.array(residue_index)
    b_factors = np.array(b_factors)
    mask = atom_mask[:, 1] * np.array([r != 'X' for r in seq])
    result = {
        'seq': seq,
        'mask': mask,
        'aatype': aatype,
        'atom_mask': atom_mask,
        'atom_positions': atom_positions,
        'residue_index': residue_index,
        'b_factors': b_factors,
    }
    return result
