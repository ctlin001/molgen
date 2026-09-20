# Example data

Structure files are not committed — see `.gitignore`. Put your own here, or fetch a test case:

```bash
# holo structure with a bound ligand (needed for pocket + interaction context)
wget https://files.rcsb.org/download/1IEP.pdb -O receptor.pdb

# extract the reference ligand for box derivation and RMSD
python - << 'PY'
from molgen.docking.rmsd import load_reference_ligand
from rdkit import Chem
mol = load_reference_ligand("receptor.pdb", resname="STI")   # imatinib
Chem.SDWriter("reference_ligand.sdf").write(mol)
PY
```

Then:

```bash
molgen run --config configs/ligand.yaml
```

Any holo PDB or mmCIF works. Apo structures run, but the pocket description will be empty and the docking box must be set explicitly in config.
