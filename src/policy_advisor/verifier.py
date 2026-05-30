"""
verifier.py — TLC Model Checker Wrapper
=========================================
Runs TLC on a generated TLA+ spec and returns structured ViolationInfo objects.

TLC invocation priority:
  1. tlc_jar argument  (explicit path to tla2tools.jar)
  2. TLC_JAR env var   (export TLC_JAR=/path/to/tla2tools.jar)
  3. VSCode TLA+ extension auto-discovery
  4. `tlc2` binary     (if TLC is installed system-wide)

TLC stdout is parsed for:
  - Invariant violations:  "Invariant <Name> is violated."
  - Property violations:   "Property <Name> is violated." / "Temporal property..."
  - Deadlock:              "Deadlock reached."
  - Clean exit:            "Model checking completed. No error has been found."

Usage:
    verifier = TLCVerifier(tlc_jar="/path/to/tla2tools.jar")
    result = verifier.run(tla_path, cfg_path)
    if not result.success:
        for v in result.violations:
            print(v.name, v.kind)
"""

import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


# ── Regex patterns for TLC output parsing ────────────────────────────────────

_RE_INVARIANT = re.compile(r"Invariant (\w+) is violated\.")
_RE_PROPERTY  = re.compile(r"(?:Property|Temporal property) (\w+) is violated\.")
_RE_OK        = re.compile(r"Model checking completed\. No error has been found\.")
_RE_DEADLOCK  = re.compile(r"Deadlock reached\.")


# ── Result data structures ────────────────────────────────────────────────────

@dataclass
class ViolationInfo:
    """One TLC-detected violation."""
    kind: str       # "invariant" | "property" | "deadlock"
    name: str       # invariant/property name, or "Deadlock"
    excerpt: str    # ~300 chars of TLC output around the violation line


@dataclass
class TLCResult:
    """Full result of a TLC run."""
    success: bool                              # True iff no violations found
    violations: list[ViolationInfo] = field(default_factory=list)
    raw_output: str = ""                       # full TLC stdout+stderr


# ── Verifier ─────────────────────────────────────────────────────────────────

class TLCVerifier:
    def __init__(self, tlc_jar: str | None = None):
        """
        Args:
            tlc_jar: explicit path to tla2tools.jar.
                     Falls back to TLC_JAR, VSCode extension auto-discovery,
                     then `tlc2` binary.
        """
        self.tlc_jar = self._resolve_tlc_jar(tlc_jar)

    @staticmethod
    def _resolve_tlc_jar(tlc_jar: str | None = None) -> str | None:
        """Resolve tla2tools.jar from explicit arg, env var, or VSCode extension."""
        candidates: list[Path] = []

        for raw in (tlc_jar, os.environ.get("TLC_JAR")):
            if raw:
                candidates.append(Path(raw).expanduser())

        vscode_ext = Path.home() / ".vscode" / "extensions"
        candidates.extend(vscode_ext.glob("tlaplus.vscode-ide-*/tools/tla2tools.jar"))
        candidates.extend(vscode_ext.glob("tlaplus.vscode-ide-*/out/tools/tla2tools.jar"))

        existing = [p.resolve() for p in candidates if p.exists()]
        if not existing:
            return None

        # Prefer the newest installed extension jar.
        return str(max(existing, key=lambda p: p.stat().st_mtime))

    # ── Command construction ──────────────────────────────────────────────────

    def _build_cmd(self, tla_path: Path, cfg_path: Path) -> list[str]:
        if self.tlc_jar:
            return [
                "java", "-jar", self.tlc_jar,
                "-config", str(cfg_path),
                str(tla_path),
            ]
        # Fallback: assume `tlc2` is on PATH
        return ["tlc2", "-config", str(cfg_path), str(tla_path)]

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, tla_path, cfg_path) -> TLCResult:
        """
        Run TLC on the given spec and return a structured TLCResult.

        The process runs in tla_path.parent so EXTENDS resolution finds
        the co-located sagallm_content_logic.tla without -lib flags.

        Raises:
            RuntimeError: if TLC binary/jar is not found or times out.
        """
        tla_path = Path(tla_path).resolve()
        cfg_path = Path(cfg_path).resolve()

        # Remove TTrace files and states from previous runs so TLC always
        # performs fresh model checking rather than trace replay.
        for f in tla_path.parent.glob("*TTrace*"):
            f.unlink(missing_ok=True)
        states_dir = tla_path.parent / "states"
        if states_dir.exists():
            import shutil
            shutil.rmtree(states_dir, ignore_errors=True)

        cmd = self._build_cmd(tla_path, cfg_path)

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(tla_path.parent),
                timeout=120,
            )
        except FileNotFoundError:
            raise RuntimeError(
                "TLC not found. Set TLC_JAR=/path/to/tla2tools.jar, install the VSCode TLA+ extension, or install tlc2."
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("TLC timed out after 120 seconds.")

        raw = proc.stdout + proc.stderr
        if proc.returncode != 0 and not self._has_known_result(raw):
            excerpt = raw[-2000:] if raw else "(no TLC output)"
            raise RuntimeError(f"TLC failed before model-checking completed:\n{excerpt}")
        return self._parse(raw)

    # ── Output parsing ────────────────────────────────────────────────────────

    def _parse(self, raw: str) -> TLCResult:
        violations = []

        for m in _RE_INVARIANT.finditer(raw):
            violations.append(ViolationInfo(
                kind="invariant",
                name=m.group(1),
                excerpt=self._excerpt(raw, m.start()),
            ))

        for m in _RE_PROPERTY.finditer(raw):
            violations.append(ViolationInfo(
                kind="property",
                name=m.group(1),
                excerpt=self._excerpt(raw, m.start()),
            ))

        dl_pos = raw.find("Deadlock reached.")
        if dl_pos != -1:
            violations.append(ViolationInfo(
                kind="deadlock",
                name="Deadlock",
                excerpt=self._excerpt(raw, dl_pos),
            ))

        success = bool(_RE_OK.search(raw)) and not violations
        return TLCResult(success=success, violations=violations, raw_output=raw)

    @staticmethod
    def _has_known_result(raw: str) -> bool:
        """Return True if TLC output contains a parseable model-checking result."""
        return bool(
            _RE_OK.search(raw)
            or _RE_INVARIANT.search(raw)
            or _RE_PROPERTY.search(raw)
            or _RE_DEADLOCK.search(raw)
        )

    @staticmethod
    def _excerpt(raw: str, pos: int, window: int = 2000) -> str:
        """Return up to `window` chars of context around `pos`.

        TLC prints the full counterexample state trace below the violation
        header, so a large window is needed to capture state variable dumps.
        """
        start = max(0, pos - 50)
        end   = min(len(raw), pos + window)
        return raw[start:end]
