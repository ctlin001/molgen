"""Prompt construction for both generation modes.

Prompts assemble deterministic context (pocket composition, interaction
fingerprint, optional SAR exemplars) and ask for strict JSON back. Every
property claim the model makes is recomputed with RDKit downstream — the
prompt asks for them only because stating a target keeps output in range.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from textwrap import dedent
from typing import Any

JSON_CONTRACT = dedent(
    """
    Reply with a single JSON object and nothing else. No markdown fences.

    {
      "molecules": [
        {
          "smiles": "<SMILES string>",
          "name": "<short identifier>",
          "rationale": "<= 40 words on why this fits the pocket>",
          "key_features": ["<pharmacophore or motif>", "..."]
        }
      ]
    }
    """
).strip()

PROPERTY_FOCUS = {
    "high_affinity": (
        "Maximise shape and electrostatic complementarity: pair donors with the "
        "acceptor-rich residues listed, fill hydrophobic subpockets, and place an "
        "aromatic ring where stacking contacts were observed."
    ),
    "selectivity": (
        "Exploit residues unique to this pocket rather than conserved family motifs. "
        "Add steric bulk adjacent to positions that differ between paralogues."
    ),
    "blood_brain_barrier": (
        "Target MW < 450, cLogP 1-3, TPSA < 70 A^2, HBD <= 2, and avoid permanent charge."
    ),
    "oral_bioavailability": (
        "Full Lipinski compliance, rotatable bonds <= 8, TPSA < 120 A^2, and avoid "
        "known metabolic soft spots such as unhindered anilines and thiophenes."
    ),
    "fragment": (
        "Rule-of-three fragment space: MW < 300, cLogP <= 3, HBD <= 3, HBA <= 3, "
        "with a clear vector for growth."
    ),
}


@dataclass
class PromptContext:
    """Everything deterministic that a prompt can draw on."""

    pocket_description: str
    interaction_description: str
    target_id: str = "unknown"
    known_actives: list[str] | None = None
    avoid_smiles: list[str] | None = None
    extra_constraints: str | None = None


def _actives_block(smiles: Iterable[str] | None, limit: int = 10) -> str:
    items = list(smiles or [])[:limit]
    if not items:
        return ""
    listed = "\n".join(f"  - {s}" for s in items)
    return f"\nKnown actives against this target (design analogues, do not copy):\n{listed}\n"


def _avoid_block(smiles: Iterable[str] | None, limit: int = 20) -> str:
    items = list(smiles or [])[:limit]
    if not items:
        return ""
    listed = "\n".join(f"  - {s}" for s in items)
    return (
        "\nAlready generated in earlier rounds. Do not repeat these or produce "
        f"close analogues of them:\n{listed}\n"
    )


def build_ligand_prompt(
    ctx: PromptContext,
    n_molecules: int = 10,
    property_focus: str | None = None,
    diversity_hint: bool = True,
) -> str:
    """Prompt for de novo small-molecule ligand generation."""
    focus = PROPERTY_FOCUS.get(property_focus, "") if property_focus else ""

    diversity = ""
    if diversity_hint:
        diversity = dedent(
            """
            Diversity requirement (this is scored downstream and enforced):
            - Every molecule must have a DIFFERENT Bemis-Murcko scaffold. Varying
              substituents on one shared ring system counts as one molecule, not many.
            - Use at least three distinct ring systems across the set.
            - Do not emit the same SMILES twice in any form, including
              resonance or kekulisation variants.
            """
        ).strip()

    return dedent(
        f"""
        Design {n_molecules} novel small molecules for the target below.

        === TARGET AND POCKET ===
        {ctx.pocket_description}

        === INTERACTION FINGERPRINT ===
        {ctx.interaction_description}
        {_actives_block(ctx.known_actives)}{_avoid_block(ctx.avoid_smiles)}
        === DESIGN BRIEF ===
        - Drug-like and synthetically tractable in under 6 steps from catalogue material.
        - Place hydrogen-bond donors and acceptors to complement the residues named above.
        - Match the hydrophobic/polar balance of the pocket.
        - No reactive or PAINS-flagged motifs: no epoxides, Michael acceptors,
          acyl halides, aldehydes, quinones, nitro on electron-poor arenes.
        {focus}

        {diversity}

        === OUTPUT FORMAT ===
        {JSON_CONTRACT}
        """
    ).strip()


def build_protac_prompt(
    ctx: PromptContext,
    warhead_smiles: str,
    e3_ligand_smiles: str,
    n_linkers: int = 10,
    anchor_distance: float | None = None,
) -> str:
    """Prompt for PROTAC linker design.

    The warhead and E3 ligand are fixed; only the linker is generated. Attachment
    points are marked ``[*]`` and are capped before RDKit embedding
    (see :func:`molgen.chem.protac.cap_wildcards`).
    """
    distance_line = (
        f"- The linker must span roughly {anchor_distance:.1f} A between anchor atoms.\n"
        if anchor_distance
        else ""
    )

    return dedent(
        f"""
        Design {n_linkers} linkers for a PROTAC against the target below.

        === TARGET AND POCKET (warhead side) ===
        {ctx.pocket_description}

        === FIXED COMPONENTS — DO NOT MODIFY ===
        Warhead (target binder): {warhead_smiles}
        E3 ligase ligand:        {e3_ligand_smiles}

        === LINKER BRIEF ===
        - Return the LINKER ONLY, as SMILES with exactly two attachment points
          written as [*]. One connects to the warhead, one to the E3 ligand.
        - Typical productive length is 8-16 heavy atoms along the shortest path.
        {distance_line}- Vary chemotype across the set: PEG, alkyl, piperazine/piperidine,
          triazole, amide, and rigid aryl or spiro spacers. Do not return
          {n_linkers} PEGs of different length.
        - Keep total PROTAC properties plausible: ternary-complex chemical space
          tolerates MW up to ~1100 and cLogP up to ~7, but flag anything beyond.
        - Avoid metabolically labile motifs: esters, unhindered benzylic amines,
          and more than one basic centre.

        === OUTPUT FORMAT ===
        {JSON_CONTRACT}

        In "key_features", state the linker class and the heavy-atom path length.
        """
    ).strip()


def build_refinement_prompt(
    base_prompt: str,
    previous: list[dict[str, Any]],
    feedback: str,
    n_molecules: int = 10,
) -> str:
    """Append a feedback block for an iterative round.

    ``feedback`` is generated programmatically from docking and validation
    results (see :func:`molgen.pipeline.summarise_round`), not written by the LLM.
    """
    lines = ["", "=== PREVIOUS ROUND ===", ""]
    for mol in previous[:15]:
        score = mol.get("docking_score")
        score_text = f"{score:.2f} kcal/mol" if isinstance(score, (int, float)) else "not docked"
        lines.append(f"  {mol.get('smiles', '?')} -> {score_text}; {mol.get('note', '')}".rstrip())

    lines += [
        "",
        "=== FEEDBACK ===",
        feedback,
        "",
        f"Design {n_molecules} improved molecules that address the feedback. "
        "Keep what worked; change what did not. New scaffolds, not re-substitutions.",
    ]
    return base_prompt + "\n".join(lines)
