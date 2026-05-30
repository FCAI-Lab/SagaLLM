"""
mapping/loader.py
=================
Loads tlc_to_opa.yaml and exposes typed helpers used by verifier.py
and guard_coordinator.py.

Usage
-----
    from policy_advisor.mapping.loader import MappingLoader

    m = MappingLoader()

    # Is this invariant fixable via Rego?
    m.is_opa_fixable("DataIsolation")        # True
    m.is_opa_fixable("AtomicTermination")    # False

    # Which Rego fix to apply?
    m.decide_fix_type(
        invariant  = "DataIsolation",
        in_E       = True,               # bad_sender ∈ E.upstream(receiver)
        bad_sender = "OI",
    )
    # → "p_cont_fix"

    # Which Rego field and action to use?
    m.get_fix_spec("DataIsolation", "p_tran_fix")
    # → {"rego_field": "allowed_transfers", "action": "remove_receiver", ...}

    # Get regex pattern string for a TLC extraction target
    m.get_pattern("receiver")
    # → '"([^"]+)"\\s*\\|->\\s*\\{[^}]*"sensitive_data"[^}]*\\}'
"""

import re
from functools import cached_property
from pathlib import Path

import yaml


_MAPPING_FILE = Path(__file__).parent / "tlc_to_opa.yaml"


class MappingLoader:
    """
    Thin wrapper around tlc_to_opa.yaml.
    Provides typed, named accessors so callers never parse YAML directly.
    """

    def __init__(self, mapping_path: str | Path = _MAPPING_FILE):
        self._path = Path(mapping_path)
        with self._path.open() as f:
            self._data = yaml.safe_load(f)

    # ── Invariant queries ─────────────────────────────────────────────────────

    def is_opa_fixable(self, invariant: str) -> bool:
        """Return True if this invariant violation can be fixed via Rego."""
        entry = self._invariants.get(invariant)
        return bool(entry and entry.get("opa_fixable", False))

    def fix_target(self, invariant: str) -> str:
        """Return the fix target for a non-OPA-fixable invariant."""
        entry = self._invariants.get(invariant, {})
        return entry.get("fix_target", "unknown")

    def action_on_violation(self, invariant: str) -> str:
        """Return developer-facing message for non-OPA-fixable violations."""
        entry = self._invariants.get(invariant, {})
        return entry.get("action_on_violation", "")

    # ── Fix type decision ─────────────────────────────────────────────────────

    def decide_fix_type(
        self,
        invariant:  str,
        in_E:       bool | None,    # bad_sender ∈ E.upstream(receiver)
        bad_sender: str | None,
    ) -> str:
        """
        Return the fix type string: "p_tran_fix" | "p_cont_fix" | "none".

        Parameters
        ----------
        invariant  : TLC invariant name (e.g. "DataIsolation")
        in_E       : True if bad_sender is an upstream of receiver in E,
                     False if not, None if unknown (multiple upstreams)
        bad_sender : the identified source agent, or None
        """
        if not self.is_opa_fixable(invariant):
            return "none"

        decision = self._invariants[invariant].get("fix_decision", {})

        if bad_sender is None or in_E is None:
            return decision.get("bad_sender_unknown", {}).get("fix_type", "p_tran_fix")

        key = "in_E_true" if in_E else "in_E_false"
        return decision.get(key, {}).get("fix_type", "p_tran_fix")

    def get_fix_spec(self, invariant: str, fix_type: str) -> dict:
        """Return the fix specification dict for a given invariant + fix_type."""
        fixes = self._invariants.get(invariant, {}).get("fixes", {})
        return fixes.get(fix_type, {})

    # ── TLC pattern access ────────────────────────────────────────────────────

    def get_pattern(self, pattern_name: str) -> str:
        """Return the raw regex string for a named TLC extraction pattern."""
        entry = self._data.get("tlc_patterns", {}).get(pattern_name, {})
        return entry.get("regex", "")

    def compile_pattern(self, pattern_name: str, flags: int = re.DOTALL) -> re.Pattern:
        """Return a compiled regex for a named TLC extraction pattern."""
        return re.compile(self.get_pattern(pattern_name), flags)

    def pattern_captures(self, pattern_name: str) -> list[str]:
        """Return the capture group names for a named pattern."""
        entry = self._data.get("tlc_patterns", {}).get(pattern_name, {})
        return entry.get("captures", [])

    # ── Rego field reference ──────────────────────────────────────────────────

    def rego_field_info(self, field_name: str) -> dict:
        """Return metadata for a Rego field (layer, type, automatable, etc.)."""
        return self._data.get("rego_fields", {}).get(field_name, {})

    # ── TLA+ model configuration ──────────────────────────────────────────────

    @property
    def threat_module_name(self) -> str:
        """Module name of the threat model spec."""
        return self._data["tla_model"]["spec_module"]

    @property
    def checked_invariants(self) -> list[str]:
        """Invariants that TLC should check (subset fixable by Rego)."""
        return self._data["tla_model"]["checked_invariants"]

    # ── Internal ──────────────────────────────────────────────────────────────

    @cached_property
    def _invariants(self) -> dict:
        return self._data.get("invariants", {})
