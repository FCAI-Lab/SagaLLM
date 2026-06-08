"""
visualize_scenarios.py — Coop Navi 시나리오 시각화

각 seed별 초기 에이전트/랜드마크 배치와
OPA 적용 전후 정보 흐름 차단을 함께 시각화합니다.

Usage:
    cd SafeSagaLLM_extension
    python experiments/coop_navi/visualize_scenarios.py
    python experiments/coop_navi/visualize_scenarios.py --seeds 42 100 200
    python experiments/coop_navi/visualize_scenarios.py --seeds 42 --show-opa
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

_ROOT = Path(__file__).parent.parent.parent
_DATASET = _ROOT / "dataset" / "coop_navi"
sys.path.insert(0, str(_DATASET))

AGENT_SIZE = 0.15
COLLISION_DIST = 0.30
AGENT_COLORS = ["#4C72B0", "#DD8452", "#55A868"]
LANDMARK_COLOR = "#C44E52"
GRID_COLOR = "#EEEEEE"


# ── 환경 파싱 ──────────────────────────────────────────────────────────────────

def get_initial_positions(seed: int, n_agents: int = 3, n_landmarks: int = 3) -> dict:
    """mpe2 환경에서 초기 위치 추출."""
    from coop_navi_adapter import run_episode, _parse_observation
    from mpe2 import simple_spread_v3

    env = simple_spread_v3.parallel_env(
        N=n_agents, max_cycles=1, continuous_actions=False, render_mode=None
    )
    observations, _ = env.reset(seed=seed)
    agent_names = list(observations.keys())

    positions = {}
    for name, obs in observations.items():
        parsed = _parse_observation(obs, n_agents, n_landmarks)
        positions[name] = {
            "pos": parsed["position"],
            "vel": parsed["velocity"],
            "landmarks": parsed["landmark_positions"],
        }

    # landmark 절대 위치 (agent_0 기준)
    landmark_positions = positions[agent_names[0]]["landmarks"]
    env.close()
    return {
        "agents": {name: positions[name]["pos"] for name in agent_names},
        "landmarks": {f"landmark_{i}": lp for i, lp in enumerate(landmark_positions)},
        "agent_names": agent_names,
    }


# ── 단일 시나리오 그리기 ────────────────────────────────────────────────────────

def draw_scenario(ax, pos_data: dict, seed: int, show_opa: bool = False):
    """하나의 subplot에 시나리오를 그린다."""
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.set_facecolor("#F8F8F8")

    # 격자
    for v in np.arange(-1.0, 1.1, 0.5):
        ax.axhline(v, color=GRID_COLOR, linewidth=0.8, zorder=0)
        ax.axvline(v, color=GRID_COLOR, linewidth=0.8, zorder=0)

    # 월드 경계
    rect = mpatches.Rectangle((-1, -1), 2, 2, linewidth=1.5,
                               edgecolor="#AAAAAA", facecolor="none", zorder=1)
    ax.add_patch(rect)

    agents = pos_data["agents"]
    landmarks = pos_data["landmarks"]
    agent_names = pos_data["agent_names"]

    # ── 충돌 위험 선 (에이전트 간 거리 < 0.6) ─────────────────────────────────
    for i, a1 in enumerate(agent_names):
        for j, a2 in enumerate(agent_names):
            if j <= i:
                continue
            p1, p2 = np.array(agents[a1]), np.array(agents[a2])
            dist = np.linalg.norm(p1 - p2)
            if dist < 0.6:
                alpha = max(0.0, 1.0 - dist / 0.6)
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                        color="#FF4444", linewidth=2.5 * alpha,
                        alpha=alpha, linestyle="--", zorder=2)
                mid = (p1 + p2) / 2
                ax.text(mid[0], mid[1] + 0.05, f"{dist:.2f}",
                        ha="center", va="bottom", fontsize=6.5,
                        color="#FF4444", alpha=0.85)

    # ── OPA 정보 차단 표시 ──────────────────────────────────────────────────────
    if show_opa:
        # Coordinator → 각 플래너에게 타 에이전트 좌표 차단 표시
        coord_center = np.array([0.0, 0.0])  # Coordinator는 중앙에 가상 배치
        for i, a_name in enumerate(agent_names):
            p = np.array(agents[a_name])
            # 자신 이외 에이전트로의 정보 흐름 차단 (점선 X)
            for j, other in enumerate(agent_names):
                if i == j:
                    continue
                other_p = np.array(agents[other])
                ax.annotate(
                    "", xy=p, xytext=other_p,
                    arrowprops=dict(
                        arrowstyle="->", color="#FF8800",
                        lw=1.2, linestyle="dotted",
                        connectionstyle="arc3,rad=0.2",
                    ),
                    zorder=3,
                )
            ax.text(p[0], p[1] - 0.22, "🚫 타 좌표\n차단됨",
                    ha="center", va="top", fontsize=5.5,
                    color="#FF8800",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec="#FF8800", alpha=0.8))

    # ── 랜드마크 ────────────────────────────────────────────────────────────────
    for lname, lpos in landmarks.items():
        ax.plot(lpos[0], lpos[1], marker="*", markersize=14,
                color=LANDMARK_COLOR, zorder=5,
                markeredgecolor="white", markeredgewidth=0.8)
        ax.text(lpos[0], lpos[1] + 0.12, lname.replace("landmark_", "L"),
                ha="center", va="bottom", fontsize=7,
                color=LANDMARK_COLOR, fontweight="bold")

    # ── 에이전트 ────────────────────────────────────────────────────────────────
    for i, a_name in enumerate(agent_names):
        p = agents[a_name]
        color = AGENT_COLORS[i]

        # 충돌 반경 원 (반투명)
        circ = mpatches.Circle(p, AGENT_SIZE, color=color, alpha=0.15, zorder=3)
        ax.add_patch(circ)
        circ_edge = mpatches.Circle(p, AGENT_SIZE, fill=False,
                                    edgecolor=color, linewidth=1.2,
                                    linestyle="--", alpha=0.6, zorder=4)
        ax.add_patch(circ_edge)

        # 에이전트 점
        ax.plot(p[0], p[1], "o", markersize=10, color=color, zorder=6,
                markeredgecolor="white", markeredgewidth=1.0)

        label = a_name.replace("agent_", "A")
        ax.text(p[0], p[1] + 0.17, label,
                ha="center", va="bottom", fontsize=8,
                color=color, fontweight="bold",
                path_effects=[pe.withStroke(linewidth=2, foreground="white")])

        # 좌표 표시
        coord_str = f"({p[0]:.2f}, {p[1]:.2f})"
        ax.text(p[0], p[1] - 0.17, coord_str,
                ha="center", va="top", fontsize=6,
                color=color, alpha=0.85,
                path_effects=[pe.withStroke(linewidth=1.5, foreground="white")])

    # ── 거리 범례 ────────────────────────────────────────────────────────────────
    # 가장 가까운 에이전트 쌍 거리
    min_dist = float("inf")
    for i, a1 in enumerate(agent_names):
        for j, a2 in enumerate(agent_names):
            if j <= i:
                continue
            d = np.linalg.norm(
                np.array(agents[a1]) - np.array(agents[a2])
            )
            min_dist = min(min_dist, d)

    risk_color = "#FF4444" if min_dist < COLLISION_DIST else (
        "#FF8800" if min_dist < 0.5 else "#55A868"
    )
    risk_label = "HIGH" if min_dist < COLLISION_DIST else (
        "MED" if min_dist < 0.5 else "LOW"
    )

    ax.set_title(
        f"seed={seed}   min_dist={min_dist:.2f}   collision_risk={risk_label}",
        fontsize=9, color=risk_color, fontweight="bold", pad=6,
    )
    ax.set_xlabel("x", fontsize=8)
    ax.set_ylabel("y", fontsize=8)
    ax.tick_params(labelsize=7)


# ── OPA 정보 흐름 다이어그램 ────────────────────────────────────────────────────

def draw_opa_flow(ax, n_agents: int = 3):
    """OPA 적용 전후 정보 흐름 DAG 다이어그램."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_facecolor("white")
    ax.set_title("SafeSagaLLM DAG & OPA 정보 차단", fontsize=10, fontweight="bold", pad=8)

    nodes = {
        "Env\nState\nAgent":    (5.0, 9.0),
        "Coordinator\nAgent":   (5.0, 7.0),
        "A0\nPlanner":          (2.0, 5.0),
        "A1\nPlanner":          (5.0, 5.0),
        "A2\nPlanner":          (8.0, 5.0),
        "Oversight\nAgent":     (5.0, 2.5),
    }
    node_colors = {
        "Env\nState\nAgent":    "#C44E52",
        "Coordinator\nAgent":   "#4C72B0",
        "A0\nPlanner":          "#4C72B0",
        "A1\nPlanner":          "#DD8452",
        "A2\nPlanner":          "#55A868",
        "Oversight\nAgent":     "#8172B2",
    }

    # 노드 그리기
    for name, (x, y) in nodes.items():
        color = node_colors[name]
        circ = mpatches.FancyBboxPatch(
            (x - 0.8, y - 0.6), 1.6, 1.2,
            boxstyle="round,pad=0.1",
            facecolor=color, edgecolor="white",
            linewidth=1.5, alpha=0.85, zorder=3,
        )
        ax.add_patch(circ)
        ax.text(x, y, name, ha="center", va="center",
                fontsize=7.5, color="white", fontweight="bold", zorder=4)

    # 엣지 그리기
    def arrow(src, dst, color="#555555", label="", blocked=False):
        x1, y1 = nodes[src]
        x2, y2 = nodes[dst]
        style = "dotted" if blocked else "solid"
        ec = "#FF4444" if blocked else color
        ax.annotate(
            "", xy=(x2, y2 + 0.6), xytext=(x1, y1 - 0.6),
            arrowprops=dict(
                arrowstyle="-|>", color=ec, lw=1.5,
                linestyle=style,
                connectionstyle="arc3,rad=0.0",
            ),
            zorder=2,
        )
        if label:
            mx = (x1 + x2) / 2 + 0.3
            my = (y1 + y2) / 2
            fc = "#FFE8E8" if blocked else "#E8F4E8"
            ec2 = "#FF4444" if blocked else "#55A868"
            ax.text(mx, my, label, ha="left", va="center",
                    fontsize=6, color=ec2,
                    bbox=dict(boxstyle="round,pad=0.2",
                              fc=fc, ec=ec2, alpha=0.9))

    # 정상 엣지
    arrow("Env\nState\nAgent", "Coordinator\nAgent",
          label="전체 좌표\n(허용)", color="#4C72B0")

    arrow("Coordinator\nAgent", "A0\nPlanner",
          label="A0좌표만\n(OPA 차단)", blocked=True)
    arrow("Coordinator\nAgent", "A1\nPlanner", blocked=True)
    arrow("Coordinator\nAgent", "A2\nPlanner", blocked=True)

    arrow("A0\nPlanner", "Oversight\nAgent", color="#4C72B0")
    arrow("A1\nPlanner", "Oversight\nAgent", color="#DD8452")
    arrow("A2\nPlanner", "Oversight\nAgent", color="#55A868")

    # 범례
    legend_items = [
        mpatches.Patch(color="#C44E52", label="Sensitive Agent (좌표 보유)"),
        mpatches.Patch(color="#4C72B0", label="Agent Planner"),
        mpatches.Patch(color="#8172B2", label="Oversight Agent"),
        plt.Line2D([0], [0], color="#FF4444", lw=1.5,
                   linestyle="dotted", label="OPA 차단 (타 에이전트 좌표)"),
        plt.Line2D([0], [0], color="#4C72B0", lw=1.5,
                   label="허용 전송"),
    ]
    ax.legend(handles=legend_items, loc="lower left",
              fontsize=7, framealpha=0.9)


# ── 메인 ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Coop Navi 시나리오 시각화")
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[0, 1, 2, 42, 100],
                        help="시각화할 seed 목록")
    parser.add_argument("--show-opa", action="store_true",
                        help="OPA 차단 화살표 오버레이 표시")
    parser.add_argument("--save", type=Path, default=None,
                        help="저장 경로 (없으면 화면 표시)")
    args = parser.parse_args()

    seeds = args.seeds
    n = len(seeds)

    # 레이아웃: 시나리오 n개 + OPA DAG 1개
    n_cols = min(n, 3)
    n_rows = (n + n_cols - 1) // n_cols + 1   # +1 for OPA DAG row

    fig = plt.figure(figsize=(5 * n_cols, 5 * n_rows))
    fig.patch.set_facecolor("white")

    fig.suptitle(
        "Coop Navi × SafeSagaLLM\n"
        "초기 배치 시나리오  (★ = 랜드마크 목표,  ● = 에이전트,  점선원 = 충돌반경 0.15)",
        fontsize=12, fontweight="bold", y=0.98,
    )

    print(f"[viz] {n}개 seed 로딩 중...")
    for idx, seed in enumerate(seeds):
        print(f"  seed={seed} ...", end=" ", flush=True)
        try:
            pos_data = get_initial_positions(seed)
            row = idx // n_cols
            col = idx % n_cols
            ax = fig.add_subplot(n_rows, n_cols, row * n_cols + col + 1)
            draw_scenario(ax, pos_data, seed, show_opa=args.show_opa)
            print("ok")
        except Exception as exc:
            print(f"⚠️  {exc}")

    # OPA DAG는 마지막 행 중앙에
    ax_dag = fig.add_subplot(n_rows, 1, n_rows)
    draw_opa_flow(ax_dag)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"\n[viz] saved → {args.save}")
    else:
        plt.show()

    print("[viz] done")


if __name__ == "__main__":
    main()
