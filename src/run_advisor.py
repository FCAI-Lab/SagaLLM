#!/usr/bin/env python3
"""
run_advisor.py — SafeSagaLLM Guard Coordinator CLI
==================================================
Entry point for the TLC → Rego auto-fix loop.

Reads pipeline.yaml, generates Rego + TLA+ specs, runs TLC, and iteratively
patches the policy until all invariants hold or a manual fix is required.
On success, automatically starts the OPA server with the verified Rego policy.

Usage:
    python run_advisor.py pipeline.yaml
    python run_advisor.py pipeline.yaml --tlc-jar /path/to/tla2tools.jar
    python run_advisor.py pipeline.yaml --max-iter 10 --no-opa

Prerequisites:
    1. TLC available:
         export TLC_JAR=/path/to/tla2tools.jar   (recommended)
         -- or -- install tlc2 on PATH
    2. sagallm_content_logic.tla present at:
         experiments/1. TLA+/core/sagallm_content_logic.tla
    3. Python packages:
         pip install pyyaml colorama
    4. opa binary on PATH (for automatic OPA startup)
"""

import argparse
import socket
import subprocess
import sys
import time
from pathlib import Path

from colorama import Fore, Style

# Add src/ to Python path so policy_advisor imports resolve without installation
sys.path.insert(0, str(Path(__file__).parent / "src"))

from policy_advisor.advisor import GuardCoordinator

_OPA_ADDR = "localhost"
_OPA_PORT = 8181


def _opa_is_running() -> bool:
    """Return True if something is already listening on OPA's port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex((_OPA_ADDR, _OPA_PORT)) == 0


def _start_opa(rego_path: Path) -> subprocess.Popen | None:
    """
    Launch OPA as a background process serving rego_path on :8181.

    Returns the Popen handle on success, None if opa binary is not found.
    """
    try:
        proc = subprocess.Popen(
            ["opa", "run", "--server", "--addr", f":{_OPA_PORT}", str(rego_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return None

    # Wait up to 5 seconds for OPA to become reachable
    for _ in range(10):
        time.sleep(0.5)
        if _opa_is_running():
            return proc

    proc.terminate()
    return None


def _handle_opa_startup(rego_path: Path) -> None:
    """
    Check port, ask user if already running, then start OPA if needed.
    Prints result; does not exit on failure (OPA is not required to run advisor).
    """
    print(Fore.CYAN + f"\n  [opa]      checking port {_OPA_PORT}..." + Style.RESET_ALL)

    if _opa_is_running():
        # Something is already on :8181 — ask whether to restart
        print(Fore.YELLOW + f"  [opa]      port {_OPA_PORT} already in use." + Style.RESET_ALL)
        answer = input("             Restart OPA with the new Rego? [y/N] ").strip().lower()
        if answer != "y":
            print(Fore.YELLOW + "  [opa]      keeping existing OPA process." + Style.RESET_ALL)
            return

        # Kill whatever is on that port
        subprocess.run(
            ["pkill", "-f", f"opa.*{_OPA_PORT}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(1.0)

    print(Fore.CYAN + f"  [opa]      starting OPA with {rego_path.name}..." + Style.RESET_ALL)
    proc = _start_opa(rego_path)

    if proc is None:
        print(
            Fore.YELLOW
            + "  [opa]      'opa' binary not found or failed to start.\n"
            + f"             Start manually:  opa run --server --addr :{_OPA_PORT} {rego_path}"
            + Style.RESET_ALL
        )
        return

    print(
        Fore.GREEN
        + f"  [opa]      running on :{_OPA_PORT}  (pid {proc.pid})\n"
        + f"             policy: {rego_path}"
        + Style.RESET_ALL
    )


def main():
    parser = argparse.ArgumentParser(
        description="SafeSagaLLM Policy Advisor — auto-verify and fix Rego via TLC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "pipeline",
        help="Path to pipeline.yaml",
    )
    parser.add_argument(
        "--tlc-jar",
        default=None,
        metavar="PATH",
        help="Path to tla2tools.jar (overrides TLC_JAR env var)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=5,
        metavar="N",
        help="Maximum fix iterations before giving up (default: 5)",
    )
    parser.add_argument(
        "--root",
        default=".",
        metavar="DIR",
        help="Project root for resolving output paths in pipeline.yaml (default: .)",
    )
    parser.add_argument(
        "--no-opa",
        action="store_true",
        help="Skip automatic OPA server startup after verification",
    )

    args = parser.parse_args()

    pipeline_path = Path(args.pipeline)
    if not pipeline_path.exists():
        print(f"Error: pipeline file not found: {pipeline_path}", file=sys.stderr)
        sys.exit(1)

    root = Path(args.root)

    # Run TLC verification + Rego auto-fix loop
    advisor = GuardCoordinator(tlc_jar=args.tlc_jar, max_iterations=args.max_iter)
    success = advisor.run(pipeline_path=pipeline_path, root_dir=root)

    if not success:
        sys.exit(1)

    # Verification passed — start OPA unless suppressed
    if not args.no_opa:
        import yaml
        pipeline_data = yaml.safe_load(pipeline_path.read_text())
        rego_rel = pipeline_data.get("rego_output", "src/policies/generated.rego")
        rego_path = (root / rego_rel).resolve()
        _handle_opa_startup(rego_path)

    sys.exit(0)


if __name__ == "__main__":
    main()
