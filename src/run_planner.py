#!/usr/bin/env python3
"""
run_planner.py — SafeSagaLLM Unified CLI
=========================================
Single entry point for all SafeSagaLLM operations.

Mode 1: 자연어 입력 (--prompt)
  자연어 → yaml 생성 → TLC 검증 → OPA 시작 → 에이전트 실행

Mode 2: yaml 직접 지정 (--yaml)
  yaml → TLC 검증 → OPA 시작 → 에이전트 실행

각 단계는 옵션으로 개별 제어 가능.

Usage:
    # Mode 1: 자연어 입력
    python src/run_planner.py --prompt "의료 파이프라인. 환자 정보는 의사만 수신 가능."

    # Mode 2: yaml 직접 지정
    python src/run_planner.py --yaml experiments/my_pipeline.yaml

    # 에이전트 실행까지
    python src/run_planner.py --prompt "..." --run

    # 검증만 (에이전트 실행 없음)
    python src/run_planner.py --yaml experiments/my_pipeline.yaml --no-run

    # 전체 출력
    python src/run_planner.py --prompt "..." --run --show-outputs --report
"""

import argparse
import socket
import subprocess
import sys
import time
from pathlib import Path

import yaml as _yaml
from colorama import Fore, Style

sys.path.insert(0, str(Path(__file__).parent))

from policy_advisor.advisor import GuardCoordinator

_OPA_ADDR = "localhost"
_OPA_PORT = 8181

# ── OPA helpers ───────────────────────────────────────────────────────────────

def _opa_is_running() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex((_OPA_ADDR, _OPA_PORT)) == 0


def _start_opa(rego_path: Path) -> subprocess.Popen | None:
    try:
        proc = subprocess.Popen(
            ["opa", "run", "--server", "--addr", f":{_OPA_PORT}", str(rego_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return None
    for _ in range(10):
        time.sleep(0.5)
        if _opa_is_running():
            return proc
    proc.terminate()
    return None


def _handle_opa_startup(rego_path: Path) -> None:
    print(Fore.CYAN + f"\n  [opa]      checking port {_OPA_PORT}..." + Style.RESET_ALL)
    if _opa_is_running():
        print(Fore.YELLOW + f"  [opa]      port {_OPA_PORT} already in use." + Style.RESET_ALL)
        answer = input("             Restart OPA with the new Rego? [y/N] ").strip().lower()
        if answer != "y":
            print(Fore.YELLOW + "  [opa]      keeping existing OPA process." + Style.RESET_ALL)
            return
        subprocess.run(
            ["pkill", "-f", f"opa.*{_OPA_PORT}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SafeSagaLLM Unified CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── 입력 모드 (둘 중 하나 필수) ───────────────────────────────────────────
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--prompt",
        metavar="TEXT",
        help="자연어로 파이프라인 설명 (yaml 자동 생성)",
    )
    input_group.add_argument(
        "--yaml",
        metavar="PATH",
        help="기존 pipeline.yaml 경로 직접 지정",
    )

    # ── yaml 생성 옵션 (--prompt 전용) ────────────────────────────────────────
    parser.add_argument(
        "--output",
        default="experiments/generated_pipeline.yaml",
        metavar="PATH",
        help="생성된 yaml 저장 경로 (기본: experiments/generated_pipeline.yaml)",
    )
    parser.add_argument(
        "--max-plan-rounds",
        type=int,
        default=3,
        metavar="N",
        help="LLM yaml 생성 재시도 횟수 (기본: 3)",
    )

    # ── 검증 옵션 ─────────────────────────────────────────────────────────────
    parser.add_argument(
        "--tlc-jar",
        default=None,
        metavar="PATH",
        help="tla2tools.jar 경로 (생략 시 TLC_JAR 또는 VSCode TLA+ extension에서 자동 탐색)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=5,
        metavar="N",
        help="GuardCoordinator 반복 횟수 (기본: 5)",
    )
    parser.add_argument(
        "--root",
        default=".",
        metavar="DIR",
        help="프로젝트 루트 경로 (기본: .)",
    )

    # ── 실행 옵션 ─────────────────────────────────────────────────────────────
    parser.add_argument(
        "--no-opa",
        action="store_true",
        help="OPA 서버 시작 생략",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="검증 후 에이전트 실행",
    )
    parser.add_argument(
        "--show-outputs",
        action="store_true",
        help="각 에이전트 LLM 출력 표시 (--run 필요)",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="OPA 감사 리포트 + D_actual 표시 (--run 필요)",
    )

    args = parser.parse_args()
    root = Path(args.root)

    # ── Step 1: yaml 확보 ─────────────────────────────────────────────────────
    if args.prompt:
        from planning_agent.input_generator import PipelinePlanner
        planner   = PipelinePlanner(max_rounds=args.max_plan_rounds)
        yaml_path = planner.run(user_prompt=args.prompt, output_path=Path(args.output))
        if yaml_path is None:
            print(Fore.RED + "\n  ❌ yaml 생성 실패." + Style.RESET_ALL)
            sys.exit(1)
    else:
        yaml_path = Path(args.yaml)
        if not yaml_path.exists():
            print(f"Error: yaml 파일을 찾을 수 없습니다: {yaml_path}", file=sys.stderr)
            sys.exit(1)

    # ── Step 2: TLC 검증 + Rego 자동 수정 ────────────────────────────────────
    coordinator = GuardCoordinator(tlc_jar=args.tlc_jar, max_iterations=args.max_iter)
    success     = coordinator.run(pipeline_path=yaml_path, root_dir=root)
    if not success:
        sys.exit(1)

    # ── Step 3: OPA 서버 시작 ─────────────────────────────────────────────────
    pipeline_data = _yaml.safe_load(yaml_path.read_text())
    rego_rel      = pipeline_data.get("rego_output", "src/policies/generated.rego")
    rego_path     = (root / rego_rel).resolve()

    if not args.no_opa:
        _handle_opa_startup(rego_path)

    # ── Step 4: 에이전트 실행 ─────────────────────────────────────────────────
    if args.run:
        from run_pipeline import load_and_run
        load_and_run(
            pipeline_path=yaml_path,
            enforce_opa=not args.no_opa,
            report=args.report,
            show_outputs=args.show_outputs,
        )

    sys.exit(0)


if __name__ == "__main__":
    main()
