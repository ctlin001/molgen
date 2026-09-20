# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/). Versioning is semantic.

## [2.0.0]

Restructured from a flat script collection into an installable package. The driver was the v1 benchmark: 1,000 molecules, 100% Lipinski-compliant, 43% duplicates, one scaffold.

### Added
- **Diversity enforcement** (`molgen.chem.diversity`) — InChIKey deduplication, Bemis-Murcko scaffold reporting (specific and generic), scaffold entropy, Tanimoto distribution, internal diversity, and a pass/fail verdict.
- **Novelty filter** — Tanimoto screen against a reference library, run before docking so compute goes to novel chemotypes.
- **MaxMin selection** — diverse subset when capping candidates for docking, replacing first-N truncation.
- **PROTAC mode** (`molgen.chem.protac`) — anchor selection, `GetShortestPath` linker identification, `FragmentOnBonds` excision, assembly, linker classification, degrader-appropriate property window.
- **Hungarian RMSD** (`molgen.docking.rmsd`) — symmetry-aware atom matching via `scipy.optimize.linear_sum_assignment`.
- **Local inference** — Ollama backend (Gemma 3 12B) as default; OpenAI retained for baseline comparison.
- **Structural alerts** — PAINS A/B/C and Brenk catalogs, plus QED and SA score.
- **Derived docking boxes** (`molgen.docking.box`) — from reference ligand, co-crystal ligand, or pocket residues, with provenance and size warnings.
- **Fragment provenance** (`molgen.chem.fragments`) — BRICS and RECAP with exact parent atom mapping.
- **Refinement rounds** — feedback generated programmatically from measured results.
- **Reports** — `run.json`, `results.csv`, self-contained HTML with embedded SVG depictions.
- **CLI** — `run`, `decompose`, `diversity`, `check`.
- **Typed YAML configuration** with validation before any work starts.
- Test suite, CI, architecture and troubleshooting docs.

### Changed
- Package layout moved to `src/molgen/` with a `molgen` entry point.
- Properties are always recomputed by RDKit; LLM-stated values are discarded.
- Deduplication moved from raw SMILES matching to InChIKey.
- Pocket detection uses a KD-tree instead of an O(n·m) double loop.
- Aromatic stacking uses named ring-atom centroids rather than whole-residue centroids.
- Parallel docking uses `ThreadPoolExecutor` (fixes `BrokenProcessPool` under Jupyter on Windows).
- Receptor preparation requires a real tool; the simplified PDB→PDBQT copy was removed as it silently corrupted scores.
- mmCIF receptors converted automatically via gemmi, with a Biopython fallback.

### Fixed
- `[*]` attachment atoms no longer reach embedding: neither MMFF nor UFF can type them, so minimisation was being skipped silently. Capping is now enforced.
- Linker excision no longer cuts ring bonds when an anchor sits inside a ring (`AtomKekulizeException`).
- Fragment atom mapping uses `fragsMolAtomMapping` instead of post-hoc substructure matching, which returned false hits on symmetric groups.
- Molecule highlighting uses the stable `Draw.MolToImage` API; `DrawMoleculeWithHighlights` broke on colour-value types across RDKit builds.
- LLM completions wrapped in markdown fences, prefixed with prose, or containing unbalanced braces are now recovered.

## [1.0.0]

Initial pipeline, adapting the DrugReAlign multi-source prompt approach from drug repurposing to de novo generation.

- PDB pocket extraction, PLIP-style interaction analysis, prompt construction
- OpenAI GPT generation, RDKit validation, Lipinski filtering
- AutoDock Vina docking, analysis notebook
