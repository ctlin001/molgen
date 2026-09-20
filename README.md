# MolGen

Pocket-conditioned molecule generation. A local LLM proposes candidate structures from a binding-site description; RDKit and AutoDock Vina decide whether they are real, drug-like, diverse, and plausible binders.

Two modes: **ligand** (de novo small molecules) and **protac** (linker design between a fixed warhead and E3 ligand).

<p align="center">
  <img src="docs/assets/molgen-pipeline.svg" width="900" alt="MolGen pipeline diagram: seven stages left to right — Pocket (BioPython, 5 Å cutoff), Prompt, Generate (Gemma 3 12B via Ollama, the only LLM stage), Validate (RDKit filters), Diversity (InChIKey dedup, Bemis–Murcko scaffolds, novelty screen), Dock (AutoDock Vina), and Report (HTML/CSV/JSON). Ligand and PROTAC-linker modes branch after the Prompt stage. A scaffold-diversity gate of at least 0.30 sits before docking, and a refinement loop feeds measured docking results back to the Prompt stage. A band beneath shows that every stage except Generate is deterministic; the LLM emits text only.">
</p>

## The one rule this project is built around

**LLMs handle language and intent. Deterministic tools handle structural chemistry.**

The model emits text. RDKit parses it, computes every property, and rejects what does not hold up. No LLM output reaches a chemist or a docking run unvalidated, and no LLM is asked to fragment, reassemble, or characterise a structure. Mixing these roles is what produces invalid SMILES, phantom molecular weights, and tables that disagree with their own source data.

## Why v2 exists

v1 generated 1,000 molecules against an aspirin benchmark. They were 100% Lipinski-compliant — and 43% exact duplicates, sharing **one** Bemis–Murcko scaffold across the entire set. Every molecule was a substituent variation on a single aromatic ring.

Property compliance is not diversity, and a pipeline that only measures compliance will report success while producing nothing useful. v2 makes diversity a first-class, enforced, measured property:

| | v1 | v2 |
|---|---|---|
| Backend | OpenAI GPT (cloud) | Gemma 3 12B via Ollama (local) |
| Deduplication | raw SMILES string match | InChIKey |
| Scaffold diversity | not measured | measured, gated, reported |
| Novelty | none | Tanimoto screen before docking |
| Candidate selection | first N | MaxMin diverse subset |
| Structural alerts | none | PAINS + Brenk |
| RMSD | Vina's `rmsd_lb`/`rmsd_ub` (misread) | Hungarian-matched, vs reference |
| Modes | ligand | ligand + PROTAC |

## Install

```bash
conda env create -f environment.yaml
conda activate molgen

ollama serve
ollama pull gemma3:12b

molgen check
```

`molgen check` reports which of RDKit, BioPython, SciPy, Vina, Open Babel and Ollama are reachable before you waste a run finding out.

## Use

```bash
# ligand generation
molgen run --config configs/ligand.yaml

# quick pass, no docking
molgen run -r data/1abc.pdb --reference data/ref.sdf -n 30 --no-docking

# property-focused, with docking-informed refinement rounds
molgen run -c configs/ligand.yaml --focus blood_brain_barrier --rounds 3

# PROTAC linkers
molgen run --config configs/protac.yaml

# GPU docking through WSL2
molgen run --config configs/vina-gpu.yaml
```

Utilities:

```bash
molgen decompose "CC(=O)Oc1ccccc1C(=O)O" --method brics   # fragment with provenance
molgen diversity runs/my_run/results.csv                  # audit any SMILES set
```

Python API:

```python
from molgen import Config, Pipeline

config = Config.from_yaml("configs/ligand.yaml")
config.generation.n_molecules = 40
artifacts = Pipeline(config).run()

print(artifacts.diversity_report["verdict"])
print(artifacts.run_dir / "report.html")
```

## Output

```
runs/<run_name>/
├── prompt.txt           exact prompt sent to the model
├── run.json             full machine-readable record
├── results.csv          one row per candidate, all properties + scores
├── report.html          self-contained, emailable, structures embedded
├── docking_box.json     derived box with provenance
├── receptor.pdbqt
├── ligands/*.sdf        3D structures
└── docking/*.pdbqt      poses
```

Every run is reproducible from `prompt.txt` plus the config block in `run.json`.

## Reading the numbers

**Docking scores** (kcal/mol) rank poses. They are not affinities and correlate poorly with measured K<sub>d</sub>. Use them to triage, not to predict potency.

| Score | Reading |
|---|---|
| < −9 | Strong; check the pose is not an artifact of an oversized box |
| −7 to −9 | Promising |
| −5 to −7 | Moderate |
| > −5 | Weak |

**RMSD.** Vina's `rmsd_lb` / `rmsd_ub` columns compare each pose to the best pose *of the same run* — they measure pose diversity, not accuracy. Accuracy needs a reference pose, which is what `molgen.docking.rmsd` computes, with Hungarian atom matching so symmetric groups are not penalised for atom ordering. Below 2.0 Å against a crystal pose is the conventional redocking success threshold.

**Diversity.** `scaffold_diversity` is unique scaffolds over unique molecules. Below 0.30, or one scaffold covering more than 40% of the set, the run is flagged FAIL regardless of how good the property distributions look.

## Traps worth knowing

Each of these cost real debugging time and is encoded in the tests.

- **PROTAC wildcards.** Generated linkers carry `[*]` attachment atoms. They parse. ETKDG will even embed them. But neither MMFF nor UFF has a parameter for atom type `*_`, so minimisation is silently skipped and the molecule travels on with unrefined geometry before Vina rejects it. Cap first: `cap_wildcards()`.
- **Fragment provenance.** Map fragment atoms to the parent with `FragmentOnBonds(..., fragsMolAtomMapping=...)`, never with post-hoc `GetSubstructMatches`. On a molecule with two identical methoxy groups, substructure matching returns both and cannot say which one produced the fragment — so the chemist edits the wrong half of the molecule.
- **Ring anchors.** Excising a linker by cutting the bond immediately next to an anchor opens a ring when the anchor sits inside one, leaving aromatic flags on non-ring atoms and raising `AtomKekulizeException`. Walk the path to the first acyclic bond.
- **Duplicates.** Deduplicate on InChIKey. The same molecule has many valid SMILES spellings, so raw string matching under-reports duplication — which is part of why v1's rate looked acceptable until it was measured properly.
- **Windows parallelism.** Use `ThreadPoolExecutor`, not `ProcessPoolExecutor`. Vina runs in a subprocess so the GIL is not the constraint, and process pools raise `BrokenProcessPool` under Jupyter on Windows.
- **Receptor prep.** A hand-rolled PDB→PDBQT copy loses partial charges and atom typing, and every score afterwards is quietly wrong. Use ADFR's `prepare_receptor` or Open Babel; this project refuses to guess.

## Layout

```
src/molgen/
├── config.py          typed config, YAML-backed
├── pipeline.py        stage orchestration
├── cli.py             run / decompose / diversity / check
├── pocket/            receptor parsing, pocket extraction, interaction fingerprint
├── generate/          LLM backends, prompt templates, completion parsing
├── chem/              validation, diversity, fragments, PROTAC handling
├── docking/           Vina wrapper, box derivation, Hungarian RMSD
└── report/            CSV and self-contained HTML
```

See [docs/architecture.md](docs/architecture.md) for the stage-by-stage contract and [docs/troubleshooting.md](docs/troubleshooting.md) for error-message-to-fix mapping.

## Development

```bash
make install     # editable install with dev extras
make test        # unit tests (no Vina or LLM required)
make lint
```

Tests that need external services are marked `requires_vina` / `requires_llm` / `slow` and skipped by default.

## Attribution

The multi-source prompt construction follows the approach in DrugReAlign (Wei et al., *BMC Biology* 22:226, 2024), redirected from repurposing approved drugs to generating novel structures. PROTAC linker handling follows the AIMLinker approach to anchor selection and linker excision.

## License

MIT — see [LICENSE](LICENSE).
