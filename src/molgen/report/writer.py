"""Run reports: CSV for analysis, self-contained HTML for sharing.

The HTML embeds structure depictions as base64 SVG so a single file can be
emailed to a chemist with no server, no assets folder, and no broken images.
"""

from __future__ import annotations

import base64
import csv
import html
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

CSS = """
:root { --fg:#1a1a1a; --muted:#666; --line:#e2e2e2; --good:#0a7d4f; --bad:#b3261e; --bg:#fff; }
* { box-sizing: border-box; }
body { font-family: ui-sans-serif, -apple-system, Segoe UI, Roboto, sans-serif;
       margin: 0 auto; padding: 32px; max-width: 1100px; color: var(--fg); background: var(--bg); }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 16px; margin: 32px 0 12px; padding-bottom: 6px; border-bottom: 1px solid var(--line); }
.sub { color: var(--muted); font-size: 13px; margin-bottom: 24px; }
.cards { display: flex; flex-wrap: wrap; gap: 12px; }
.card { border: 1px solid var(--line); border-radius: 8px; padding: 12px 16px; min-width: 140px; }
.card .v { font-size: 20px; font-weight: 600; }
.card .k { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--line); }
th { font-weight: 600; color: var(--muted); font-size: 11px; text-transform: uppercase; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.smiles { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 11px; word-break: break-all; }
.good { color: var(--good); } .bad { color: var(--bad); }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 14px; }
.mol { border: 1px solid var(--line); border-radius: 8px; padding: 8px; text-align: center; }
.mol img { width: 100%; height: auto; }
.mol .cap { font-size: 11px; color: var(--muted); margin-top: 4px; }
.verdict { padding: 10px 14px; border-radius: 6px; font-size: 13px; margin: 8px 0 16px; }
.verdict.pass { background: #eaf6f0; color: var(--good); }
.verdict.fail { background: #fdecea; color: var(--bad); }
@media (prefers-color-scheme: dark) {
  :root { --fg:#e8e8e8; --muted:#9a9a9a; --line:#333; --bg:#151515; }
  .mol img { background:#fff; border-radius:4px; }
}
"""


def svg_for_smiles(smiles: str, size: tuple[int, int] = (280, 220)) -> str | None:
    """Base64 data URI of a 2D depiction."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit.Chem.Draw import rdMolDraw2D

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        AllChem.Compute2DCoords(mol)
        drawer = rdMolDraw2D.MolDraw2DSVG(*size)
        drawer.drawOptions().clearBackground = False
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()
    except Exception as exc:
        log.debug("depiction failed for %s: %s", smiles, exc)
        return None


def write_csv(artifacts, path: Path) -> Path:
    """Flat table of every candidate with properties and docking score."""
    scores = {d["ligand_id"]: d for d in artifacts.docking}

    columns = [
        "name", "canonical_smiles", "valid", "passes_all", "mw", "logp", "hbd", "hba",
        "tpsa", "rotatable_bonds", "aromatic_rings", "fraction_csp3", "qed", "sa_score",
        "lipinski_violations", "alerts", "best_affinity", "reference_rmsd",
    ]

    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for cand in artifacts.candidates:
            props = cand.get("properties") or {}
            key = cand.get("inchikey", "")[:14]
            docking = scores.get(key) or scores.get(cand.get("name", "")) or {}
            writer.writerow({
                "name": cand.get("name") or key,
                "canonical_smiles": cand.get("canonical_smiles"),
                "valid": cand.get("valid"),
                "passes_all": cand.get("passes_all"),
                **{k: props.get(k) for k in (
                    "mw", "logp", "hbd", "hba", "tpsa", "rotatable_bonds",
                    "aromatic_rings", "fraction_csp3", "qed", "sa_score")},
                "lipinski_violations": "; ".join(cand.get("lipinski_violations") or []),
                "alerts": "; ".join(cand.get("alerts") or []),
                "best_affinity": docking.get("best_affinity"),
                "reference_rmsd": docking.get("reference_rmsd"),
            })
    return path


def _card(label: str, value: Any) -> str:
    return f'<div class="card"><div class="v">{html.escape(str(value))}</div><div class="k">{html.escape(label)}</div></div>'


def write_html(artifacts, path: Path, top_n: int = 24) -> Path:
    """Self-contained HTML summary."""
    cfg = artifacts.config
    meta = artifacts.pocket.get("metadata", {})
    val = artifacts.validation_summary or {}
    divr = artifacts.diversity_report or {}

    scores = {d["ligand_id"]: d for d in artifacts.docking}
    rows = []
    for cand in artifacts.candidates:
        if not cand.get("valid"):
            continue
        key = cand.get("inchikey", "")[:14]
        docking = scores.get(key) or scores.get(cand.get("name", "")) or {}
        rows.append({**cand, "_docking": docking})

    rows.sort(key=lambda r: (
        r["_docking"].get("best_affinity") if r["_docking"].get("best_affinity") is not None else 0
    ))
    top = rows[:top_n]

    verdict = divr.get("verdict", "")
    verdict_class = "pass" if verdict.startswith("PASS") else "fail"

    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>MolGen report - {html.escape(cfg.get('run_name', 'run'))}</title>",
        f"<style>{CSS}</style></head><body>",
        f"<h1>MolGen run: {html.escape(cfg.get('run_name', 'run'))}</h1>",
        f"<div class='sub'>{html.escape(cfg.get('mode', ''))} mode &middot; "
        f"target {html.escape(str(meta.get('id', 'unknown')))} &middot; "
        f"backend {html.escape(cfg.get('generation', {}).get('backend', ''))}</div>",
        "<div class='cards'>",
        _card("generated", val.get("n", 0)),
        _card("valid", val.get("valid", 0)),
        _card("validity", f"{val.get('validity_rate', 0):.0%}"),
        _card("Ro5 pass", val.get("ro5_pass", 0)),
        _card("mean QED", val.get("mean_qed", 0)),
        _card("scaffolds", divr.get("n_scaffolds", 0)),
        _card("docked", len(artifacts.docking)),
        "</div>",
    ]

    if verdict:
        parts += [
            "<h2>Diversity</h2>",
            f"<div class='verdict {verdict_class}'>{html.escape(verdict)}</div>",
            "<table><tr><th>metric</th><th>value</th></tr>",
        ]
        for key in (
            "n_unique", "duplicate_rate", "n_scaffolds", "n_generic_scaffolds",
            "scaffold_diversity", "scaffold_entropy", "largest_scaffold_share",
            "internal_diversity", "mean_pairwise_tanimoto",
        ):
            if key in divr:
                parts.append(
                    f"<tr><td>{key.replace('_', ' ')}</td>"
                    f"<td class='num'>{html.escape(str(divr[key]))}</td></tr>"
                )
        parts.append("</table>")

    parts += ["<h2>Top candidates</h2>", "<div class='grid'>"]
    for row in top[:12]:
        uri = svg_for_smiles(row.get("canonical_smiles", ""))
        affinity = row["_docking"].get("best_affinity")
        caption = f"{row.get('name') or ''}"
        if affinity is not None:
            caption += f" &middot; {affinity:.2f} kcal/mol"
        image = f"<img alt='structure' src='{uri}'>" if uri else "<em>no depiction</em>"
        parts.append(f"<div class='mol'>{image}<div class='cap'>{caption}</div></div>")
    parts.append("</div>")

    parts += [
        "<h2>Results table</h2><table>",
        "<tr><th>name</th><th>SMILES</th><th>MW</th><th>cLogP</th><th>TPSA</th>"
        "<th>QED</th><th>SA</th><th>affinity</th><th>RMSD</th><th>alerts</th></tr>",
    ]
    for row in top:
        props = row.get("properties") or {}
        docking = row["_docking"]
        affinity = docking.get("best_affinity")
        rmsd = docking.get("reference_rmsd")
        alerts = row.get("alerts") or []
        parts.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('name') or ''))}</td>"
            f"<td class='smiles'>{html.escape(str(row.get('canonical_smiles') or ''))}</td>"
            f"<td class='num'>{props.get('mw', '')}</td>"
            f"<td class='num'>{props.get('logp', '')}</td>"
            f"<td class='num'>{props.get('tpsa', '')}</td>"
            f"<td class='num'>{props.get('qed', '')}</td>"
            f"<td class='num'>{props.get('sa_score', '') if props.get('sa_score') is not None else ''}</td>"
            f"<td class='num'>{f'{affinity:.2f}' if affinity is not None else ''}</td>"
            f"<td class='num'>{f'{rmsd:.2f}' if rmsd is not None else ''}</td>"
            f"<td class='{'bad' if alerts else 'good'}'>{len(alerts) or 'clean'}</td>"
            "</tr>"
        )
    parts.append("</table>")

    timings = artifacts.timings or {}
    if timings:
        parts.append("<h2>Timings</h2><table><tr><th>stage</th><th>seconds</th></tr>")
        for stage, seconds in timings.items():
            parts.append(f"<tr><td>{html.escape(stage)}</td><td class='num'>{seconds:.1f}</td></tr>")
        parts.append("</table>")

    parts.append(
        "<div class='sub' style='margin-top:32px'>Docking scores are Vina affinities and rank poses, "
        "not affinities. Vina rmsd_lb / rmsd_ub are pose-diversity values; the RMSD column here is "
        "computed against the reference ligand.</div>"
    )
    parts.append("</body></html>")

    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def write_reports(artifacts) -> dict[str, Path]:
    """Write every report format for one run."""
    out = {
        "csv": write_csv(artifacts, artifacts.run_dir / "results.csv"),
        "html": write_html(artifacts, artifacts.run_dir / "report.html"),
    }
    log.info("reports written to %s", artifacts.run_dir)
    return out
