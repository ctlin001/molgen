# Architecture

## The separation of concerns

One rule governs every module:

> LLMs handle language and intent. Deterministic tools handle structural chemistry.

The LLM appears in exactly one stage. It receives text and returns text. Everything upstream of it (pocket parsing, interaction fingerprinting, prompt assembly) and everything downstream (parsing, validation, diversity, docking, reporting) is deterministic and reproducible.

This is not stylistic. The failures that motivated it were concrete:

- LLMs asked to fragment or reassemble molecules produce valence errors and unparseable SMILES.
- LLMs asked to report molecular weight report it confidently and wrongly.
- LLMs asked to generate structured tables produce tables that disagree with their own source data — which is why, in the sibling IND-document work, tables are injected programmatically from source rather than generated.

The model is good at proposing chemotypes given a described pocket. It is not an arithmetic engine or a chemistry toolkit, and the architecture does not ask it to be one.

## Stage contracts

Each stage takes a defined input, produces a defined output, and can be run in isolation.

### 1. Pocket — `molgen.pocket`

**In:** receptor PDB or mmCIF. **Out:** `PocketFeatures`, `InteractionProfile`, two prose descriptions.

Pocket residues are every standard residue with an atom within a cutoff (default 5 Å) of a non-solvent heteroatom, found via a KD-tree. Interactions are geometric only — distance-and-element criteria for hydrophobic contacts, hydrogen bonds and aromatic stacking. No energy model.

This is context for a prompt, not a binding prediction. Ring centroids for stacking use named ring atoms rather than whole-residue centroids, which otherwise drift toward the backbone and inflate counts.

**Apo receptors:** with no ligand present, the pocket is empty and the docking box must come from config. This is logged, not silently tolerated.

### 2. Prompt — `molgen.generate.prompts`

**In:** descriptions, optional known actives, optional exclusion list. **Out:** one prompt string, persisted to `prompt.txt`.

The prompt requests strict JSON and states the diversity requirement explicitly — different Bemis–Murcko scaffolds, not different substituents. That instruction alone does not produce diversity, which is why stage 5 enforces it, but it measurably shifts the output distribution.

Property targets in the prompt keep output in range. They are never trusted; stage 4 recomputes everything.

### 3. Generate — `molgen.generate`

**In:** prompt. **Out:** `list[Candidate]` — SMILES strings and rationale, nothing verified.

Backends are pluggable behind a protocol. The default is local Ollama inference: receptor context stays on-premise and cost per run is electricity. The OpenAI backend is retained for baseline comparison.

Parsing has a three-step recovery ladder, because local models reliably do at least one of: wrap JSON in markdown fences, add a conversational preamble, drop a closing brace.

1. strip fences, parse the outermost object
2. scan for any balanced `{...}` block
3. line-wise SMILES scraping, where every token is confirmed with `Chem.MolFromSmiles` — pattern matching alone is only a prefilter

`n_calls` oversamples the same prompt at jittered temperature. Mode collapse is the dominant failure mode of a single call, and oversampling plus deduplication is the cheapest available lever against it.

### 4. Validate — `molgen.chem.validate`

**In:** candidates. **Out:** `ValidationResult` per candidate.

Fixed order: parse → sanitise → canonicalise → InChIKey → properties → Ro5 → PAINS/Brenk → 3D embed.

Every property is recomputed. Molecules under 6 heavy atoms are rejected as scraping artifacts. Thresholds differ by mode: ligand mode treats structural alerts as disqualifying; PROTAC mode records them without disqualifying, since degraders occupy a different property space by construction.

Embedding refuses molecules containing dummy atoms — see the PROTAC note below.

### 5. Diversity — `molgen.chem.diversity`

**In:** passing candidates. **Out:** deduplicated, novelty-screened, size-capped selection plus a `DiversityReport`.

The stage that exists because v1 did not have it.

- **Dedupe on InChIKey.** Raw SMILES comparison under-reports duplication because one molecule has many valid spellings.
- **Report** scaffold counts (specific and generic), scaffold entropy, largest-scaffold share, internal diversity, and the Tanimoto histogram. A verdict of FAIL is returned when duplicate rate exceeds 15%, scaffold diversity falls below 0.30, or one scaffold covers more than 40% of the set.
- **Novelty screen** against a reference library before docking. A fingerprint comparison costs microseconds; a Vina run costs seconds to minutes. Screening first means compute goes to novel chemotypes rather than to rediscovering the library.
- **MaxMin selection** when capping how many candidates proceed, so the cap preserves chemical space coverage instead of taking whichever the model emitted first.

### 6. Dock — `molgen.docking`

**In:** 3D ligands, receptor, box. **Out:** `DockingResult` per ligand.

The search box is derived, never drawn by hand — a manually drawn grid box is the least reproducible step in a docking protocol and does not survive being handed to a colleague. Preference order: reference ligand extent → co-crystal ligand → pocket residue CAs → explicit config. The box records its own provenance and warns on edges beyond 30 Å, where Vina slows sharply and pose quality degrades.

Reference RMSD uses Hungarian atom matching within each element type. Index-order RMSD on a molecule with a carboxylate or a para-substituted ring is inflated by several ångströms by atom relabeling alone — `tests/test_rmsd.py` demonstrates this on benzene.

### 7. Report — `molgen.report`

**In:** `RunArtifacts`. **Out:** `run.json`, `results.csv`, `report.html`.

HTML embeds depictions as base64 SVG so one file can be emailed with no assets folder and no broken images, and it respects the reader's dark-mode setting.

## Refinement rounds

With `rounds > 1`, feedback for the next round is generated *programmatically* from measured validation, diversity and docking numbers — `summarise_round()` in `pipeline.py`. The model does not grade its own output. Feedback names the scaffolds that scored well, the diversity verdict, and the motifs that docked weakly.

## Extending

**New backend:** implement `complete()` and a `name` property, register it in `build_backend`.

**New property filter:** add to `StructureValidator.validate` and surface the threshold in `ValidationConfig`.

**New generation mode:** add a prompt builder in `prompts.py`, a preparation hook analogous to `Pipeline._prepare_linkers`, and a config dataclass.

**New docking engine:** match the `VinaDocker` interface — `prepare_receptor`, `prepare_ligand`, `dock`, `dock_many`.

## Known limitations

- Interaction detection is geometric. It ignores hydrogen positions, angles, and desolvation, and overcounts relative to PLIP. It is prompt context, not analysis.
- Docking scores rank poses, not affinities. Anything that survives here needs MM/GBSA or FEP before it means much.
- Ensemble effects are out of scope. Molecules designed against a single conformer dock well into that conformer and often fall apart under rescoring; an ensemble-consistency filter belongs in the design from the start rather than bolted on later.
- Synthetic accessibility is the SA score only — a heuristic, not a route.
