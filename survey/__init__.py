"""The survey tool — backend-agnostic measurement for the on-device LLM inference survey.

Enumerates work from the model registry (models.yaml), spawns one backend process
per (model, variant, provider, task) cell, samples memory from the outside, and
aggregates schema-valid results. The contract (CLI + the two JSON schemas +
backend.toml) is the only coupling to a backend; see the root README.md.
"""

__version__ = "0.1.0"
