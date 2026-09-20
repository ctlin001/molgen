# Contributing

## Setup

```bash
conda env create -f environment.yaml
conda activate molgen
make install
make test
```

## The rule

Before adding anything, check it against the architectural constraint:

> LLMs handle language and intent. Deterministic tools handle structural chemistry.

A change that has an LLM fragment, reassemble, characterise, or compute a property on a molecule will be rejected. Route it through RDKit.

## Standards

- Type hints on public functions. `from __future__ import annotations` at the top of modules.
- Docstrings say *why*, not *what*. A trap or a non-obvious decision belongs in the docstring at the point where someone would otherwise reintroduce the bug.
- `ruff check src tests` clean. Line length 100.
- New behaviour comes with a test. New bug fixes come with a test that fails without the fix.

## Tests

```bash
make test        # default: skips anything needing external services
pytest -m requires_vina
pytest -m requires_llm
```

Mark anything needing Vina, Ollama, or more than a few seconds with `requires_vina`, `requires_llm`, or `slow`. CI runs only the unmarked set.

Chemistry tests should assert chemistry, not string equality. Compare InChIKeys rather than SMILES, and prefer a property assertion over a hardcoded canonical string that changes between RDKit releases.

## Adding a component

**Backend** — implement `complete()` and `name`, register in `build_backend`, add a `requires_llm` test.

**Filter** — add to `StructureValidator.validate`, surface the threshold in `ValidationConfig`, document the default in the config comment.

**Mode** — prompt builder in `prompts.py`, preparation hook in `Pipeline`, config dataclass, example YAML in `configs/`.

## Data

Never commit receptors, ligand libraries, docking output, or anything under `runs/`. `.gitignore` covers the usual cases. Internal targets and proprietary SAR data do not belong in this repository at all.
