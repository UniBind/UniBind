# UniBind

UniBind is an AI-based framework to predict Protein-Protein binding affinity with the ability of integrating multi-source and heterogenous biological datasets. The UniBind consists of three major components: protein representation as graph data structure, BindFormer blocks with geometry and energy attention (GEA), and multi-task learning for heterogeneous biological data integration. This package provides  information on deep learning models and instructions on related scripts to run UniBind. 

## Datasets
Our training dataset contains large-scale heterogenous protein-protein interation dataset curated from public sources. Datasets used for traning UniBind include SKEMPI 2.0, an open-source database with information on binding free energy changes after mutations of structurally resolved protein-protein interactions, and three other protein-protein binding affinity datasets (PADBs) constructed and curated from retrospective deep mutational scanning，flow cytometry or neutralization assay experiments. All raw data for the training is available from public sources and here are the links to download them.

* [SKEMPI v2.0](https://life.bsc.es/pid/skempi2/)
* PADB-SA
	[(Starr et al., 2020)](https://doi.org/10.1016/j.cell.2020.08.012)
	[(Starr et al., 2022)](https://doi.org/10.1101/2022.02.24.481899.)
* PADB-AS
	[(Chan et al., 2020)](https://doi.org/10.1126/science.abc0870.)
* PADB-SAb
	[(Cao et al., 2022)](https://doi.org/10.1038/s41586-021-04385-3.)
	[(Liu et al., 2022)](https://doi.org/10.1038/s41586-021-04388-0.)
	[(Wang et al., 2021)](https://doi.org/10.1016/j.immuni.2021.06.003.)

## Inference

We recommend setting up the runtime environment and run UniBind via anaconda. The following steps are required in order to run UniBind:

1. Install EvoEF2 [(Huang et al., 2020)](https://doi.org/10.1093/bioinformatics/btz740) following the installation instruction in the homepage:
```
cd softwares/EvoEF2
./build.sh
```

2. Install [Anaconda](https://www.anaconda.com/). For linux:
```
wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
bash ./Anaconda3-2022.05-Linux-x86_64.sh
```

3. Setup conda environment for UniBind via yaml file:
```
conda env create -f env.yml
```

4. Activate unibind env:
```
conda activate unibind
```

5. Run inference for given pdb and mutation list:

```
./run.py inference input/7df4.pdb input/mutation.txt model/model.pt output/ --path_bin_evoef2 softwares/EvoEF2/EvoEF2
```

## Online web service API
Apart from code for local training and inference, we also provide online web service for general binding affinity prediction and COVID-19 related prediction tasks.

**Please note that due to the computational complexity and network instability, the average response time for a request is about 1 minute.**

### 1. General binding affinity prediction
For a given complex structure in pdb format and residue mutations, our web service can predict the binding affinity change under the given mutations. You can access our web service for general binding affinity prediction with command line with `curl`:

**Command line**:

```
curl -F "mutation=C308A" -F "chainids=B" -F "pdb=@1a22.pdb" http://180.184.64.75:7999/affinity
```
This command needs to be executed with the wild type pdb file under the same directory. The pdb file is downloaded from https://www.rcsb.org/. The `mutation` parameter represents the substitution mutations from the wild type complex. The `chainids` parameter represents the the chain id where the substitution mutations taking place. The `pdb` parameter represents the wild type pdb file name. 

In this exmple, we predict the affinity change of a variant of 1a22 with a substitution mutation that replacing the Cysteine to Alanine at 308 on chain B. 

**Example output**:

```
{
	"affinity_change": Float
}
```

**Output explanation**:

`affinity_change` is the binding affinity change of the given complex variant.

### 2. COVID-19 related prediction tasks
Our web service can also provide the predictions for affinity change, antibody escape score and fitness score between Spike protein trimer-ACE2 complex. You can access our web service for COVID-19 related prediction tasks with command line with `curl` or web browser: 

**Command line**:

```
curl -F "mutation=V213G,G339D,R346T,S371L,S373P,S375F,T376A,D405N,R408S,K417N,N440K,L452R,S477N,T478K,E484A,F486V,Q493R,Q498R,N501Y,Y505H,D614G,H655Y,N658S,N679K,P681H,N764K" http://180.184.64.75:7999/covid19
```

**Web browser**:

```
http://180.184.64.75:7999/covid19?mutation=V213G,G339D,R346T,S371L,S373P,S375F,T376A,D405N,R408S,K417N,N440K,L452R,S477N,T478K,E484A,F486V,Q493R,Q498R,N501Y,Y505H,D614G,H655Y,N658S,N679K,P681H,N764K
```

**Example output**:

```
{
	"ACE2_affinity_change": Float,
	"Antibody_escape_score": Float,
	"Evo_score": Float
}
```

**Output explanation**:
`ACE2_affinity_change` is the binding affinity change between the given Spike protein trimer variant and ACE2. `Antibody_escape_score` is the average escape score of known antibodies to the given Spike protein variant. `Evo_score` is the fitness score of the given Spike protein variant.

## Reference
Cao, Y., et al. (2022). Omicron escapes the majority of existing SARS-CoV-2 neutralizing antibodies. Nature 602, 657–663.

Chan, K.K., et al. (2020). Engineering human ACE2 to optimize binding to the spike protein of SARS coronavirus 2. Science 369, 1261–1265.

Jankauskaitė, J., et al. (2019). SKEMPI 2.0: an updated benchmark of changes in protein–protein binding energy, kinetics and thermodynamics upon mutation. Bioinformatics 35, 462–469.

Liu, L., et al. (2022). Striking antibody evasion manifested by the Omicron variant of SARS-CoV-2. Nature 602, 676–681.

Starr, T.N., et al. (2020). Deep Mutational Scanning of SARS-CoV-2 Receptor Binding Domain Reveals Constraints on Folding and ACE2 Binding. Cell 182, 1295-1310.e20.

Starr, T.N., et al. (2022). Shifting mutational constraints in the SARS-CoV-2 receptor-binding domain during viral evolution. 2022.02.24.481899.

Wang, R., et al. (2021). Analysis of SARS-CoV-2 variant mutations reveals neutralization escape mechanisms and the ability to use ACE2 receptors from additional species. Immunity 54, 1611-1621.e5.

Huang, X., Pearce, R., and Zhang, Y. (2020). EvoEF2: accurate and fast energy function for computational protein design. Bioinformatics 36, 1135-1142.

