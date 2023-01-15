#!/usr/bin/env python3
import fire
from pathlib import Path
import shutil
from subprocess import getoutput


def pdb_mutate(path_pdb, path_mutation, path_output, path_bin='EvoEF2'):
    path_pdb = Path(path_pdb)
    path_output = Path(path_output)
    path_bin = Path(path_bin)
    path_output.parent.mkdir(parents=True, exist_ok=True)

    paht_pdb_temp = path_bin.parent/'input.pdb'
    paht_mut_temp = path_bin.parent/'mutation.txt'
    paht_output_temp = path_bin.parent/'input_Model_0001.pdb'

    shutil.copy(path_pdb,paht_pdb_temp)
    shutil.copy(path_mutation,paht_mut_temp)
    cmd = f'cd {path_bin.parent}; ./{path_bin.name} --command=BuildMutant --pdb input.pdb --mutant_file mutation.txt'
    msg = getoutput(cmd)
    print(msg)
    shutil.copy(paht_output_temp, path_output)
    return msg


class ProteinTools(object):
    def __init__(self):
        super(ProteinTools, self).__init__()

    def evoef2(self, path_pdb, path_mutation, path_output, path_bin='EvoEF2'):
        pdb_mutate(path_pdb, path_mutation, path_output, path_bin=path_bin)


if __name__ == '__main__':
    fire.Fire(ProteinTools)
