"""Plugin registry and configuration models.

Status: planned (no public API yet). See ``docs/design/proposal.md`` and the issue
tracker.

Scope: Namespaces prior, observation, estimator and gate; entry-point discovery;
Pydantic/YAML configuration models shaped like flepimop2's, without importing flepimop2
(the provider package does that).

Reference implementations to port (rule of two): flepimop2 module.py (pattern only).
"""

__all__: list[str] = []
