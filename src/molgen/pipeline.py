"""Pipeline orchestration.

Stages, in order:

    1. pocket    -- parse receptor, extract pocket, fingerprint interactions
    2. prompt    -- assemble deterministic context into a generation prompt
    3. generate  -- LLM proposes SMILES (ligand) or linkers (PROTAC)
    4. validate  -- RDKit: parse, properties, Ro5, alerts, 3D embed
    5. diversity -- dedupe, scaffold report, novelty screen, MaxMin subset
    6. dock      -- Vina, plus reference RMSD where a reference exists
    7. report    -- JSON + CSV + HTML

Stages 4-6 are deterministic. The LLM appears only in stage 3, and it never
touches a structure — it emits text that RDKit then accepts or rejects.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .chem import diversity as div
from .chem import protac as protac_tools
from .chem.validate import StructureValidator, ValidationResult, summarise
from .config import Config
from .docking.box import DockingBox, box_from_config, box_from_pocket, box_from_reference_ligand
from .docking.rmsd import compute_rmsd, load_reference_ligand
from .docking.vina import VinaDocker
from .generate.backends import build_backend
from .generate.generator import Candidate, MoleculeGenerator
from .generate.prompts import (
    PromptContext,
    build_ligand_prompt,
    build_protac_prompt,
    build_refinement_prompt,
)
from .pocket.interactions import InteractionAnalyzer, describe_interactions
from .pocket.parser import PocketExtractor, describe_pocket

log = logging.getLogger(__name__)


@dataclass
class RunArtifacts:
    """Everything one run produced."""

    run_dir: Path
    config: dict[str, Any] = field(default_factory=dict)
    pocket: dict[str, Any] = field(default_factory=dict)
    interactions: dict[str, Any] = field(default_factory=dict)
    prompt: str = ""
    candidates: list[dict[str, Any]] = field(default_factory=list)
    validation_summary: dict[str, Any] = field(default_factory=dict)
    diversity_report: dict[str, Any] = field(default_factory=dict)
    novelty: list[dict[str, Any]] = field(default_factory=list)
    docking: list[dict[str, Any]] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)

    def save(self) -> Path:
        path = self.run_dir / "run.json"
        payload = {
            "config": self.config,
            "pocket": self.pocket,
            "interactions": self.interactions,
            "validation_summary": self.validation_summary,
            "diversity_report": self.diversity_report,
            "novelty": self.novelty,
            "candidates": self.candidates,
            "docking": self.docking,
            "timings": self.timings,
        }
        path.write_text(json.dumps(payload, indent=2, default=str))
        return path


def _load_smiles_file(path: Path | None) -> list[str]:
    if not path or not Path(path).exists():
        return []
    path = Path(path)
    if path.suffix.lower() == ".csv":
        import csv

        with path.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            return []
        key = "smiles" if "smiles" in rows[0] else list(rows[0])[0]
        return [r[key] for r in rows if r.get(key)]
    return [
        line.split()[0]
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


def summarise_round(
    validated: list[ValidationResult],
    docking_results: list[dict[str, Any]],
    diversity_report: dict[str, Any],
) -> str:
    """Programmatic feedback text for the next generation round.

    Written from measured numbers, not by the LLM — the model does not get to
    grade its own output.
    """
    lines = []
    stats = summarise(validated)
    if stats.get("n"):
        lines.append(
            f"Validity {stats['validity_rate']:.0%}, "
            f"{stats['ro5_pass']}/{stats['valid']} Ro5-compliant, "
            f"mean QED {stats['mean_qed']}."
        )

    verdict = diversity_report.get("verdict", "")
    if verdict.startswith("FAIL"):
        lines.append(
            f"Diversity problem: {verdict}. The next set must use different ring "
            "systems, not new substituents on the same core."
        )

    scored = [d for d in docking_results if d.get("best_affinity") is not None]
    if scored:
        best = min(scored, key=lambda d: d["best_affinity"])
        mean = sum(d["best_affinity"] for d in scored) / len(scored)
        lines.append(
            f"Docking: mean {mean:.2f} kcal/mol, best {best['best_affinity']:.2f} "
            f"({best['ligand_id']}). Build on the best-scoring chemotypes."
        )
        weak = [d for d in scored if d["best_affinity"] > -6.0]
        if weak:
            lines.append(f"{len(weak)} candidates scored weaker than -6.0 kcal/mol; avoid those motifs.")

    failures = [r for r in validated if not r.valid]
    if failures:
        reasons = {r.error for r in failures if r.error}
        lines.append(f"{len(failures)} unparseable outputs ({'; '.join(list(reasons)[:3])}).")

    return "\n".join(lines) or "No measurable issues in the previous round."


class Pipeline:
    """End-to-end run driver."""

    def __init__(self, config: Config):
        problems = config.validate()
        if problems:
            raise ValueError("invalid configuration:\n  - " + "\n  - ".join(problems))
        self.cfg = config
        self.run_dir = Path(config.paths.output_dir) / config.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts = RunArtifacts(run_dir=self.run_dir, config=config.to_dict())

    # ------------------------------------------------------------ 1. pocket

    def stage_pocket(self) -> tuple[str, str, Any]:
        started = time.perf_counter()
        receptor = Path(self.cfg.paths.receptor)

        extractor = PocketExtractor(receptor, self.cfg.ligand_resname)
        residues = extractor.pocket_residues(cutoff=self.cfg.pocket_cutoff)
        features = extractor.features(residues)
        metadata = extractor.metadata()

        if not residues:
            log.warning(
                "no pocket residues found (apo structure?); the prompt will carry "
                "metadata only and the docking box must come from config"
            )

        pocket_text = describe_pocket(metadata, features)

        try:
            analyzer = InteractionAnalyzer(receptor, self.cfg.ligand_resname)
            profile = analyzer.analyse()
            interaction_text = describe_interactions(profile)
            self.artifacts.interactions = profile.to_dict()
        except Exception as exc:
            log.warning("interaction analysis failed: %s", exc)
            interaction_text = "No interaction data available."

        self.artifacts.pocket = {"metadata": metadata, "features": features.to_dict()}
        self.artifacts.timings["pocket"] = time.perf_counter() - started
        return pocket_text, interaction_text, features

    # ------------------------------------------------------------ 2. prompt

    def stage_prompt(self, pocket_text: str, interaction_text: str) -> str:
        ctx = PromptContext(
            pocket_description=pocket_text,
            interaction_description=interaction_text,
            target_id=self.artifacts.pocket.get("metadata", {}).get("id", "unknown"),
            known_actives=_load_smiles_file(self.cfg.paths.known_actives),
        )

        if self.cfg.mode == "protac":
            pc = self.cfg.protac
            warhead, e3 = pc.warhead_smiles, pc.e3_ligand_smiles

            if pc.parent_protac_smiles and pc.warhead_anchor is not None:
                excision = protac_tools.excise_linker(
                    pc.parent_protac_smiles, pc.warhead_anchor, pc.e3_anchor
                )
                warhead = warhead or excision.warhead_smiles
                e3 = e3 or excision.e3_smiles
                self.artifacts.pocket["excision"] = excision.to_dict()
                log.info(
                    "excised linker %s (path length %d)",
                    excision.linker_smiles,
                    excision.linker_path_length,
                )

            prompt = build_protac_prompt(
                ctx,
                warhead_smiles=warhead,
                e3_ligand_smiles=e3,
                n_linkers=self.cfg.generation.n_molecules,
                anchor_distance=pc.anchor_distance,
            )
        else:
            prompt = build_ligand_prompt(
                ctx,
                n_molecules=self.cfg.generation.n_molecules,
                property_focus=self.cfg.generation.property_focus,
                diversity_hint=self.cfg.generation.diversity_hint,
            )

        (self.run_dir / "prompt.txt").write_text(prompt)
        self.artifacts.prompt = prompt
        return prompt

    # ---------------------------------------------------------- 3. generate

    def stage_generate(self, prompt: str, round_index: int = 0) -> list[Candidate]:
        started = time.perf_counter()
        backend = build_backend(self.cfg.generation.backend)
        if hasattr(backend, "health_check"):
            backend.health_check()

        generator = MoleculeGenerator(
            backend,
            temperature=self.cfg.generation.temperature,
            max_tokens=self.cfg.generation.max_tokens,
        )
        candidates = generator.generate_batch(
            prompt, n_calls=self.cfg.generation.n_calls, round_index=round_index
        )

        if self.cfg.mode == "protac":
            candidates = self._prepare_linkers(candidates)

        self.artifacts.timings[f"generate_r{round_index}"] = time.perf_counter() - started
        log.info("round %d: %d raw candidates", round_index, len(candidates))
        return candidates

    def _prepare_linkers(self, candidates: list[Candidate]) -> list[Candidate]:
        """Filter linkers to two attachment points, then cap wildcards.

        Capping is what makes the linker embeddable and dockable; uncapped ``*``
        atoms fail ETKDG and are rejected by Vina.
        """
        pc = self.cfg.protac
        prepared = []
        for cand in candidates:
            ok, reason = protac_tools.validate_linker(
                cand.smiles,
                min_heavy=pc.min_linker_heavy_atoms,
                max_heavy=pc.max_linker_heavy_atoms,
            )
            if not ok:
                log.debug("rejected linker %s: %s", cand.smiles, reason)
                continue

            assembled = None
            if pc.warhead_smiles and pc.e3_ligand_smiles:
                assembled = protac_tools.assemble_protac(
                    pc.warhead_smiles, cand.smiles, pc.e3_ligand_smiles
                )

            cand.key_features = list(cand.key_features) + [
                f"linker_class={protac_tools.classify_linker(protac_tools.cap_wildcards(cand.smiles))}"
            ]
            cand.smiles = assembled or protac_tools.cap_wildcards(cand.smiles)
            prepared.append(cand)
        return prepared

    # ---------------------------------------------------------- 4. validate

    def stage_validate(self, candidates: list[Candidate]) -> list[ValidationResult]:
        started = time.perf_counter()
        validator = StructureValidator(
            max_lipinski_violations=self.cfg.validation.max_lipinski_violations,
            allow_alerts=self.cfg.validation.allow_alerts or self.cfg.mode == "protac",
            embed=self.cfg.validation.embed_3d,
            sdf_dir=self.run_dir / "ligands",
        )
        results = validator.validate_many([(c.smiles, c.name) for c in candidates])
        self.artifacts.validation_summary = summarise(results)
        self.artifacts.timings["validate"] = time.perf_counter() - started
        log.info("validation: %s", self.artifacts.validation_summary)
        return results

    # --------------------------------------------------------- 5. diversity

    def stage_diversity(self, results: list[ValidationResult]) -> list[ValidationResult]:
        if not self.cfg.diversity.enabled:
            return [r for r in results if r.passes_all]

        started = time.perf_counter()
        passing = [r for r in results if r.passes_all]
        smiles = [r.canonical_smiles for r in passing]

        report = div.analyse_diversity(smiles)
        self.artifacts.diversity_report = report.to_dict()
        log.info("diversity: %s", report.verdict())

        by_smiles = {r.canonical_smiles: r for r in passing}
        unique, _ = div.deduplicate(smiles)
        kept = [by_smiles[s] for s in unique if s in by_smiles]

        library = self.cfg.paths.novelty_library
        if library and Path(library).exists():
            novelty_filter = div.NoveltyFilter.from_file(
                library, threshold=self.cfg.diversity.novelty_threshold
            )
            annotations = novelty_filter.annotate([r.canonical_smiles for r in kept])
            self.artifacts.novelty = annotations
            novel = {a["smiles"] for a in annotations if a["novel"]}
            dropped = len(kept) - len(novel)
            kept = [r for r in kept if r.canonical_smiles in novel]
            log.info("novelty filter removed %d near-duplicates of known compounds", dropped)

        cap = self.cfg.diversity.max_to_dock
        if cap and len(kept) > cap:
            if self.cfg.diversity.maxmin_selection:
                chosen = set(div.pick_diverse_subset([r.canonical_smiles for r in kept], cap))
                kept = [r for r in kept if r.canonical_smiles in chosen]
            else:
                kept = kept[:cap]
            log.info("selected %d candidates for docking", len(kept))

        self.artifacts.timings["diversity"] = time.perf_counter() - started
        return kept

    # -------------------------------------------------------------- 6. dock

    def _build_box(self, features) -> DockingBox:
        cfg = self.cfg.docking
        if cfg.box_center and cfg.box_size:
            return box_from_config(tuple(cfg.box_center), tuple(cfg.box_size))

        reference = self.cfg.paths.reference_ligand
        if reference and Path(reference).exists():
            return box_from_reference_ligand(
                reference, padding=cfg.box_padding, cubic=cfg.cubic_box
            )

        extractor = PocketExtractor(self.cfg.paths.receptor, self.cfg.ligand_resname)
        centroid = extractor.ligand_centroid()
        if centroid:
            import numpy as np

            from .docking.box import box_from_coords

            _, ligand_atoms, _ = extractor._partition_atoms()
            coords = np.asarray([a.coord for a in ligand_atoms if a.element != "H"])
            return box_from_coords(coords, cfg.box_padding, cubic=cfg.cubic_box,
                                   source="cocrystal_ligand")

        return box_from_pocket(features, padding=cfg.box_padding, cubic=cfg.cubic_box)

    def stage_dock(self, results: list[ValidationResult], features) -> list[dict[str, Any]]:
        if not self.cfg.docking.enabled:
            log.info("docking disabled")
            return []

        started = time.perf_counter()
        cfg = self.cfg.docking
        docker = VinaDocker(
            executable=cfg.executable,
            use_wsl=cfg.use_wsl,
            exhaustiveness=cfg.exhaustiveness,
            num_modes=cfg.num_modes,
            cpu=cfg.cpu,
            seed=cfg.seed,
        )
        log.info("docking engine: %s", docker.check())

        box = self._build_box(features)
        box.save(self.run_dir / "docking_box.json")

        receptor = docker.prepare_receptor(
            self.cfg.paths.receptor, self.run_dir / "receptor.pdbqt"
        )
        ligands = [r.sdf_path for r in results if r.sdf_path]
        if not ligands:
            log.warning("no 3D ligands available to dock")
            return []

        docked = docker.dock_many(
            receptor, ligands, box, self.run_dir / "docking", max_workers=cfg.max_workers
        )

        if cfg.compute_rmsd and self.cfg.paths.reference_ligand:
            self._attach_rmsd(docked)

        self.artifacts.timings["dock"] = time.perf_counter() - started
        return [d.to_dict() for d in docked]

    def _attach_rmsd(self, docked) -> None:
        from .docking.vina import poses_to_sdf

        try:
            reference = load_reference_ligand(self.cfg.paths.reference_ligand)
        except Exception as exc:
            log.warning("could not load reference ligand for RMSD: %s", exc)
            return

        for result in docked:
            if not result.success or not result.output_path:
                continue
            try:
                sdf = poses_to_sdf(
                    result.output_path,
                    Path(result.output_path).with_suffix(".sdf"),
                    keep=1,
                )
                pose = load_reference_ligand(sdf)
                result.reference_rmsd = compute_rmsd(reference, pose).rmsd
            except Exception as exc:
                log.debug("RMSD failed for %s: %s", result.ligand_id, exc)

    # ------------------------------------------------------------------ run

    def run(self) -> RunArtifacts:
        log.info("run '%s' (%s mode) -> %s", self.cfg.run_name, self.cfg.mode, self.run_dir)
        total_started = time.perf_counter()

        pocket_text, interaction_text, features = self.stage_pocket()
        prompt = self.stage_prompt(pocket_text, interaction_text)

        all_validated: list[ValidationResult] = []
        docking_records: list[dict[str, Any]] = []
        current_prompt = prompt

        for round_index in range(self.cfg.generation.rounds):
            candidates = self.stage_generate(current_prompt, round_index)
            if not candidates:
                log.warning("round %d produced nothing usable", round_index)
                break

            validated = self.stage_validate(candidates)
            all_validated.extend(validated)

            selected = self.stage_diversity(all_validated)
            docking_records = self.stage_dock(selected, features)

            if round_index + 1 < self.cfg.generation.rounds:
                feedback = summarise_round(
                    all_validated, docking_records, self.artifacts.diversity_report
                )
                current_prompt = build_refinement_prompt(
                    prompt,
                    [
                        {
                            "smiles": d["ligand_id"],
                            "docking_score": d.get("best_affinity"),
                        }
                        for d in docking_records[:15]
                    ],
                    feedback,
                    n_molecules=self.cfg.generation.n_molecules,
                )

        self.artifacts.candidates = [r.to_dict() for r in all_validated]
        self.artifacts.docking = docking_records
        self.artifacts.timings["total"] = time.perf_counter() - total_started

        from .report.writer import write_reports

        self.artifacts.save()
        write_reports(self.artifacts)
        log.info("run complete in %.1fs", self.artifacts.timings["total"])
        return self.artifacts
