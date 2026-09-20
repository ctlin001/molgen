"""Command-line interface.

    molgen run --config configs/ligand.yaml
    molgen run --config configs/ligand.yaml --receptor data/1abc.pdb --no-docking
    molgen decompose "CC(=O)Oc1ccccc1C(=O)O" --method brics
    molgen diversity results.smi
    molgen check
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import Config

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def _setup_logging(verbosity: int) -> None:
    level = logging.WARNING if verbosity == 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt="%H:%M:%S")


# ------------------------------------------------------------------- run


def cmd_run(args: argparse.Namespace) -> int:
    from .pipeline import Pipeline

    config = Config.from_yaml(args.config) if args.config else Config()

    if args.receptor:
        config.paths.receptor = Path(args.receptor)
    if args.reference:
        config.paths.reference_ligand = Path(args.reference)
    if args.output:
        config.paths.output_dir = Path(args.output)
    if args.run_name:
        config.run_name = args.run_name
    if args.mode:
        config.mode = args.mode
    if args.backend:
        config.generation.backend = args.backend
    if args.n is not None:
        config.generation.n_molecules = args.n
    if args.rounds is not None:
        config.generation.rounds = args.rounds
    if args.temperature is not None:
        config.generation.temperature = args.temperature
    if args.focus:
        config.generation.property_focus = args.focus
    if args.no_docking:
        config.docking.enabled = False
    if args.gpu:
        config.docking.executable = args.gpu
        config.docking.use_wsl = True
    config.paths.resolve()

    problems = config.validate()
    if problems:
        print("Configuration problems:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 2

    if args.dry_run:
        print(json.dumps(config.to_dict(), indent=2))
        return 0

    artifacts = Pipeline(config).run()
    summary = artifacts.validation_summary
    print(f"\nRun complete: {artifacts.run_dir}")
    print(f"  generated {summary.get('n', 0)}, valid {summary.get('valid', 0)}, "
          f"passing all filters {summary.get('passes_all', 0)}")
    if artifacts.diversity_report:
        print(f"  diversity: {artifacts.diversity_report.get('verdict', 'n/a')}")
    scored = [d for d in artifacts.docking if d.get("best_affinity") is not None]
    if scored:
        best = min(scored, key=lambda d: d["best_affinity"])
        print(f"  best docking score: {best['best_affinity']:.2f} kcal/mol ({best['ligand_id']})")
    print(f"  report: {artifacts.run_dir / 'report.html'}")
    return 0


# ------------------------------------------------------------- decompose


def cmd_decompose(args: argparse.Namespace) -> int:
    from .chem.fragments import decompose

    result = decompose(args.smiles, method=args.method)
    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    print(f"Input: {result['input_smiles']}")
    print(f"Scaffold: {result['scaffold']['scaffold_smiles']}")
    print(f"Generic scaffold: {result['scaffold']['generic_scaffold_smiles']}")
    print(f"\nFragments ({args.method}):")
    for frag in result["fragments"]:
        print(f"  {frag['fragment_id']:>4}  {frag['smiles']:<40} {frag['label']}")
        print(f"        parent atoms: {frag['parent_atom_indices']}")
    print("\nFunctional groups:")
    for group in result["functional_groups"]:
        print(f"  {group['group']:<20} atoms {group['atom_indices']}")
    return 0


# -------------------------------------------------------------- diversity


def cmd_diversity(args: argparse.Namespace) -> int:
    from .chem.diversity import analyse_diversity

    path = Path(args.input)
    if path.suffix.lower() == ".csv":
        import csv

        with path.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        column = args.column if rows and args.column in rows[0] else "canonical_smiles"
        smiles = [r[column] for r in rows if r.get(column)]
    else:
        smiles = [
            line.split()[0]
            for line in path.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]

    report = analyse_diversity(smiles)
    print(json.dumps(report.to_dict(), indent=2))
    return 0 if report.verdict().startswith("PASS") else 1


# ------------------------------------------------------------------ check


def cmd_check(args: argparse.Namespace) -> int:
    ok = True
    print("Environment check\n")

    try:
        import rdkit

        print(f"  [ok]   rdkit {rdkit.__version__}")
    except ImportError:
        print("  [FAIL] rdkit not installed")
        ok = False

    try:
        import Bio

        print(f"  [ok]   biopython {Bio.__version__}")
    except ImportError:
        print("  [FAIL] biopython not installed")
        ok = False

    try:
        import scipy

        print(f"  [ok]   scipy {scipy.__version__}")
    except ImportError:
        print("  [FAIL] scipy not installed (required for Hungarian RMSD)")
        ok = False

    from .docking.vina import VinaDocker

    try:
        print(f"  [ok]   vina: {VinaDocker(executable=args.vina).check()}")
    except Exception as exc:
        print(f"  [warn] vina unavailable: {exc}")

    import shutil

    for tool in ("obabel", "prepare_receptor"):
        mark = "ok" if shutil.which(tool) else "warn"
        found = shutil.which(tool) or "not found"
        print(f"  [{mark}] {tool}: {found}")

    from .generate.backends import OllamaBackend

    try:
        backend = OllamaBackend(model=args.model)
        backend.health_check()
        print(f"  [ok]   ollama: {backend.model} reachable at {backend.host}")
    except Exception as exc:
        print(f"  [warn] ollama: {exc}")

    print("\n" + ("Core dependencies present." if ok else "Missing core dependencies."))
    return 0 if ok else 1


# ------------------------------------------------------------------ parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="molgen",
        description="Pocket-conditioned molecule generation with LLM + RDKit + Vina",
    )
    parser.add_argument("-v", "--verbose", action="count", default=1,
                        help="-v info (default), -vv debug, use -q for warnings only")
    parser.add_argument("-q", "--quiet", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the full pipeline")
    run.add_argument("-c", "--config", help="YAML config file")
    run.add_argument("-r", "--receptor", help="receptor PDB/CIF (overrides config)")
    run.add_argument("--reference", help="reference ligand for box and RMSD")
    run.add_argument("-o", "--output", help="output directory")
    run.add_argument("--run-name", help="name for this run's subdirectory")
    run.add_argument("--mode", choices=["ligand", "protac"])
    run.add_argument("--backend", help="e.g. ollama:gemma3:12b or openai:gpt-4o-mini")
    run.add_argument("-n", type=int, help="molecules per round")
    run.add_argument("--rounds", type=int, help="refinement rounds")
    run.add_argument("--temperature", type=float)
    run.add_argument("--focus", choices=[
        "high_affinity", "selectivity", "blood_brain_barrier",
        "oral_bioavailability", "fragment",
    ])
    run.add_argument("--no-docking", action="store_true")
    run.add_argument("--gpu", metavar="EXE",
                     help="path to AutoDock-Vina-GPU-2-1; implies WSL path translation")
    run.add_argument("--dry-run", action="store_true", help="print resolved config and exit")
    run.set_defaults(func=cmd_run)

    dec = sub.add_parser("decompose", help="fragment a molecule for the chemist UI")
    dec.add_argument("smiles")
    dec.add_argument("--method", choices=["brics", "recap"], default="brics")
    dec.add_argument("--json", action="store_true")
    dec.set_defaults(func=cmd_decompose)

    diversity = sub.add_parser("diversity", help="diversity report for a SMILES or CSV file")
    diversity.add_argument("input")
    diversity.add_argument("--column", default="canonical_smiles")
    diversity.set_defaults(func=cmd_diversity)

    check = sub.add_parser("check", help="verify the environment")
    check.add_argument("--vina", default="vina")
    check.add_argument("--model", default="gemma3:12b")
    check.set_defaults(func=cmd_check)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(0 if args.quiet else args.verbose)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        logging.getLogger("molgen").error("%s", exc, exc_info=args.verbose > 1)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
