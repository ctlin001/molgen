# Troubleshooting

Error message → cause → fix. Run `molgen check` first; it catches most of these before a run starts.

## Generation

**`cannot reach Ollama at http://localhost:11434`**
The service is not running. `ollama serve` (Linux/macOS); on Windows the installer registers it as a service — check it started. If Ollama runs on another host, set `OLLAMA_HOST`.

**`model 'gemma3:12b' not pulled`**
`ollama pull gemma3:12b`. The 12B model needs roughly 8 GB of free VRAM or system RAM; on a 6 GB card it runs partly on CPU and is slow but functional.

**`JSON parse failed; falling back to SMILES scraping`** (warning)
The model ignored the output contract. The fallback usually recovers the molecules. If it happens on most calls, lower `generation.temperature` to 0.6, or try a larger model — smaller models follow format instructions less reliably.

**Zero candidates from a call**
Check `prompt.txt`. An apo receptor produces an empty pocket description and the model has nothing to work with. Supply a holo structure or set the box explicitly.

**Every generated molecule looks the same**
Expected without the diversity stage, which is why it exists. Confirm `diversity.enabled: true`, raise `generation.n_calls`, and read the `verdict` in the report. If it still collapses, the prompt context is too thin — add `paths.known_actives`.

## Validation

**`refusing to embed a molecule with dummy atoms`**
A `[*]`-containing structure reached embedding. In PROTAC mode `_prepare_linkers` caps these; if you are calling the API directly, use `cap_wildcards()` first. This is deliberate: ETKDG embeds dummies happily, but neither MMFF nor UFF can type them, so minimisation is skipped silently and the geometry is unrefined.

**Everything fails Ro5 in PROTAC mode**
Degraders sit beyond rule-of-five by design. `configs/protac.yaml` sets `max_lipinski_violations: 4` and `allow_alerts: true`. Use `protac_property_window()` for degrader-appropriate bounds instead.

**`SA_Score module unavailable`** (warning)
RDKit's contrib directory is missing from this install. `sa_score` is reported as `null`; everything else is unaffected. Conda-forge RDKit includes it.

## Docking

**`'vina' not found`**
`conda install -c conda-forge autodock-vina`, or set `docking.executable` to the full path.

**`No receptor preparation tool found`**
Install ADFR suite (provides `prepare_receptor`) or Open Babel. The pipeline refuses to fall back on a hand-rolled PDB→PDBQT copy, because that silently drops partial charges and atom typing and corrupts every score that follows.

**`AtomKekulizeException: non-ring atom marked aromatic`**
A ring bond was cut. In `excise_linker` this is handled by walking to the first acyclic bond; if you hit it elsewhere, you are fragmenting through a ring.

**`BrokenProcessPool`**
`ProcessPoolExecutor` under Jupyter on Windows. This project uses `ThreadPoolExecutor` throughout — if you see this, custom code introduced a process pool.

**Docking scores all near −3 to −4 kcal/mol**
Usually an oversized or misplaced box. Check `docking_box.json`: the `warnings` field flags edges over 30 Å. Supply `paths.reference_ligand` so the box is derived from the real binding site.

**RMSD values of 3–6 Å from Vina output**
Those are `rmsd_lb` / `rmsd_ub` — pose diversity relative to the best pose of the same run, not accuracy against a crystal structure. Reference RMSD is the `reference_rmsd` column, computed by `molgen.docking.rmsd` against `paths.reference_ligand`.

**mmCIF receptor rejected**
Converted automatically by `cif_to_pdb`. Install `gemmi` (`pip install gemmi`) for the fast path; Biopython is the fallback.

## Environment

**Vina-GPU under WSL2**
Set `docking.use_wsl: true` and give the Linux path to the binary; Windows paths are translated to `/mnt/<drive>/...` automatically. Keep `max_workers: 1` — the GPU is the bottleneck and oversubscribing it slows the run. Build notes: `docs/vina-gpu-wsl2.md`.

**RDKit import fails after pip install**
Install RDKit from conda-forge. Pip wheels lag and omit contrib modules.

**Notebook cannot import molgen**
`pip install -e .` from the repo root, and make sure the notebook kernel is the `molgen` environment.
