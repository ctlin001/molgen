"""MolGen — pocket-conditioned molecule generation.

An LLM proposes candidate structures from a binding-pocket description; RDKit
and AutoDock Vina decide whether they are real, drug-like, diverse and
plausible binders. The separation is deliberate and load-bearing: language
models are used for intent and ideation only, never for structural chemistry.
"""

__version__ = "2.0.0"

from .config import Config
from .pipeline import Pipeline, RunArtifacts

__all__ = ["Config", "Pipeline", "RunArtifacts", "__version__"]
