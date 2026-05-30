#!/usr/bin/env python3
"""
run_tests.py — SafeSagaLLM Advisor 동작 검증 실험
===================================================
두 가지 시나리오를 순서대로 실행하여 advisor의 동작을 확인:

[실험 1] Preflight 오류 감지 (test_preflight_errors.yaml)
  - 미등록 에이전트 (Ghost Agent)
  - 사이클 (Plan Oversight → Order Setup)
  - policy의 미등록 에이전트 (Phantom Agent)
  → TLC 실행 없이 즉시 오류 위치와 수정 방법 출력

[실험 2] TLC 위반 감지 및 자동 수정 (test_tlc_violations.yaml)
  - ContentDataIsolation: "Secret Key"가 sensitive_keywords에 없음
  → TLC 위반 감지 후 guard_coordinator.py 조치 결과 출력

Usage:
    cd "/Users/jwlee/Desktop/SafeSagaLLM extension"
    python experiments/run_tests.py --tlc-jar /path/to/tla2tools.jar
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from colorama import Fore, Style
from policy_advisor.advisor import GuardCoordinator

EXPERIMENTS_DIR = Path(__file__).parent
ROOT_DIR        = EXPERIMENTS_DIR.parent


def section(title: str):
    print(Fore.CYAN + f"\n{'#'*65}")
    print(f"  {title}")
    print(f"{'#'*65}" + Style.RESET_ALL)


def run_experiment(name: str, yaml_path: Path, tlc_jar: str, max_iterations: int = 5):
    section(name)
    advisor = GuardCoordinator(tlc_jar=tlc_jar, max_iterations=max_iterations)
    success = advisor.run(pipeline_path=yaml_path, root_dir=ROOT_DIR)
    result = "PASSED" if success else "STOPPED (see above)"
    color  = Fore.GREEN if success else Fore.RED
    print(color + f"\n  결과: {result}" + Style.RESET_ALL)
    return success


def main():
    parser = argparse.ArgumentParser(
        description="SafeSagaLLM advisor 동작 검증 실험"
    )
    parser.add_argument(
        "--tlc-jar",
        default=None,
        metavar="PATH",
        help="Path to tla2tools.jar (default: auto-detect from TLC_JAR or VSCode TLA+ extension)",
    )
    args = parser.parse_args()

    print(Fore.CYAN + "\n" + "="*65)
    print("  SafeSagaLLM Advisor 동작 검증 실험")
    print("="*65 + Style.RESET_ALL)

    # ── 실험 1: Preflight 오류 ────────────────────────────────────────────────
    run_experiment(
        name="[실험 1] Preflight 오류 감지 (미등록 에이전트 / 사이클 / policy 불일치)",
        yaml_path=EXPERIMENTS_DIR / "test_preflight_errors.yaml",
        tlc_jar=args.tlc_jar,
    )

    # ── 실험 2: TLC 위반 감지 ─────────────────────────────────────────────────
    run_experiment(
        name="[실험 2] TLC 위반 감지 및 조치 (DataIsolation + ContentDataIsolation)",
        yaml_path=EXPERIMENTS_DIR / "test_tlc_violations.yaml",
        tlc_jar=args.tlc_jar,
    )

    # ── 실험 3: AgentDojo Banking ─────────────────────────────────────────────
    run_experiment(
        name="[실험 3] AgentDojo Banking — IBAN/거래내역의 Unauthorized Receiver 유출 차단",
        yaml_path=EXPERIMENTS_DIR / "test_agentdojo_banking.yaml",
        tlc_jar=args.tlc_jar,
    )

    # ── 실험 4: AgentDojo Workspace ───────────────────────────────────────────
    run_experiment(
        name="[실험 4] AgentDojo Workspace — 이메일/보안코드의 Unauthorized Forwarder 유출 차단",
        yaml_path=EXPERIMENTS_DIR / "test_agentdojo_workspace.yaml",
        tlc_jar=args.tlc_jar,
    )

    # ── 실험 5: AgentDojo Slack ───────────────────────────────────────────────
    run_experiment(
        name="[실험 5] AgentDojo Slack — 채널 메시지의 Unauthorized Publisher 유출 차단",
        yaml_path=EXPERIMENTS_DIR / "test_agentdojo_slack.yaml",
        tlc_jar=args.tlc_jar,
    )

    # ── 실험 6: AgentDojo Travel ──────────────────────────────────────────────
    run_experiment(
        name="[실험 6] AgentDojo Travel — 여권번호/계좌번호의 Unauthorized Recipient 유출 차단",
        yaml_path=EXPERIMENTS_DIR / "test_agentdojo_travel.yaml",
        tlc_jar=args.tlc_jar,
    )

    # ── 실험 7~10: P_cont 두 번째 레이어 검증 (suite별) ──────────────────────
    run_experiment(
        name="[실험 7] AgentDojo Banking P_cont — P_tran 통과 후 P_cont 차단",
        yaml_path=EXPERIMENTS_DIR / "test_agentdojo_banking_pcont.yaml",
        tlc_jar=args.tlc_jar,
    )

    run_experiment(
        name="[실험 8] AgentDojo Workspace P_cont — P_tran 통과 후 P_cont 차단",
        yaml_path=EXPERIMENTS_DIR / "test_agentdojo_workspace_pcont.yaml",
        tlc_jar=args.tlc_jar,
    )

    run_experiment(
        name="[실험 9] AgentDojo Slack P_cont — P_tran 통과 후 P_cont 차단",
        yaml_path=EXPERIMENTS_DIR / "test_agentdojo_slack_pcont.yaml",
        tlc_jar=args.tlc_jar,
    )

    run_experiment(
        name="[실험 10] AgentDojo Travel P_cont — P_tran 통과 후 P_cont 차단",
        yaml_path=EXPERIMENTS_DIR / "test_agentdojo_travel_pcont.yaml",
        tlc_jar=args.tlc_jar,
    )

    print(Fore.CYAN + "\n" + "="*65)
    print("  실험 완료")
    print("="*65 + Style.RESET_ALL)


if __name__ == "__main__":
    main()
