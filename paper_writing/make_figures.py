"""
make_figures.py — Figures for the ARC Prize 2026 Paper Track writeup.

"Chronos Solver: BFS-First Genuine Search for ARC-AGI-3 (v12 → v19)"

Every number plotted here is taken directly from the repository's own
artifacts (the v12 BFS solution caches, the version READMEs, and the
CHRONOS_EVOLUTION research note) so the charts are reproducible and honest,
not illustrative. Sources are cited inline next to each data block.

Run:
    python paper_writing/make_figures.py
Output:
    paper_writing/figures/*.png   (300 dpi, ready for the Kaggle media gallery)

The Jupyter notebook `chronos_paper_charts.ipynb` mirrors these cells so the
charts can also be viewed inline.
"""

import os
import json
import glob

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs, no display needed
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
V12_CACHE_DIR = os.path.join(
    REPO, "CommunitySolutions", "chronos_solver", "archive", "v12"
)
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# --------------------------------------------------------------------------
# House style — one palette, one look, used by every figure
# --------------------------------------------------------------------------
INK = "#1f2330"        # near-black text
MUTE = "#6b7280"       # secondary text / captions
GRID = "#e6e8ee"       # gridlines
PANEL = "#ffffff"      # panel background

ERA1 = "#b07aa1"       # Era I  — LLM orchestration (v1–v11)
ERA2 = "#3b6fb5"       # Era II — symbolic search  (v12–v13)  ← the hero
ERA3 = "#52a36a"       # Era III— model-based RL    (v14–v19)
WIN = "#2f6fb0"        # "this works" blue
LOSS = "#d9534f"       # "this failed / regressed" red
GOLD = "#e8a13a"       # highlight / frontier

plt.rcParams.update(
    {
        "figure.facecolor": PANEL,
        "axes.facecolor": PANEL,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "text.color": INK,
        "xtick.color": MUTE,
        "ytick.color": MUTE,
        "font.size": 11,
        "font.family": "DejaVu Sans",
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 120,
    }
)


def _save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=PANEL)
    plt.close(fig)
    print("  wrote", os.path.relpath(path, REPO))


def _caption(fig, text):
    fig.text(0.5, -0.02, text, ha="center", va="top", fontsize=8.5, color=MUTE)


# ==========================================================================
# REAL DATA — loaded from the repo, with documented constants for the rest.
# ==========================================================================

def load_v12_levels():
    """Levels v12 genuinely solved, read from its BFS solution caches.

    Each `v12_bfs_cache_<game>.json` maps level-index -> action list. We count
    levels per game and keep the action length per level (the search depth).
    """
    out = {}
    for f in sorted(glob.glob(os.path.join(V12_CACHE_DIR, "v12_bfs_cache_*.json"))):
        game = os.path.basename(f).replace("v12_bfs_cache_", "").replace(".json", "")
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        lvls = {}
        for k, v in d.items():
            try:
                lvls[int(k)] = len(v) if isinstance(v, (list, tuple)) else 0
            except Exception:
                pass
        if lvls:
            out[game] = lvls
    return out


V12 = load_v12_levels()  # real, from disk

# Leaderboard scores (CHRONOS_EVOLUTION_v1_to_v19.md §1; v19 regression memo).
SCORES = {
    "v12\nlive white-box BFS": (0.22, WIN),
    "black-box only\n(v18/v19 ablation)": (0.01, LOSS),
    "v19 submitted\n(plumbing regression)": (0.02, LOSS),
}

# Chaining fix: synthetic set_level() baseline vs chained L0..L(n-1) baseline.
# (v12 readme.md "Wrong BFS baseline" bug: L1 58->45, L2 97->39, L3 140->43)
CHAIN = {"L1": (58, 45), "L2": (97, 39), "L3": (140, 43)}

# World-model memorisation fix, v14 -> v15 (CHRONOS_EVOLUTION §5 v14/v15).
WM_FIX = {
    "train\naccuracy": (99.7, 99.7),
    "fresh-episode\naccuracy": (36.6, 90.0),  # v15 reported as ">90%"
}

# Honest negatives + transfer ceiling (CHRONOS_EVOLUTION §6.3).
COLOUR_AUG = {"Mac A/B": -0.02, "RTX +D4": -0.061}  # change-acc lift (disproved)
WM_PLATEAU = 0.17  # changed-pixel transfer accuracy plateau

# Version timeline (CHRONOS_EVOLUTION §2 mermaid graph).
TIMELINE = [
    ("v1", "Baseline harness + logging", ERA1),
    ("v2", "Agentic swarm, anti-oscillation", ERA1),
    ("v3", "Multimodal A*, goal extraction", ERA1),
    ("v4", "Episodic memory, HUD semantics", ERA1),
    ("v5", "Silent-reset detection", ERA1),
    ("v6", "Spatial CoT + curiosity", ERA1),
    ("v7", "Retrospectives, UI masking", ERA1),
    ("v8", "Pre-game planning, sub-goals", ERA1),
    ("v9", "Autonomous discovery, autopsies", ERA1),
    ("v10", "Offline Gemma quantization", ERA1),
    ("v11", "ADK hierarchical swarm + sandbox", ERA1),
    ("v12", "Parallel BFS  →  0.22", ERA2),
    ("v13", "Search ladder (IW/EHC/A*) + CNN", ERA2),
    ("v14", "Causal world model + VQ tokens", ERA3),
    ("v15", "Token-conditioned dynamics fix", ERA3),
    ("v16", "Generalisation curriculum (gated)", ERA3),
    ("v17", "Informed search: ForgeNet+TRM+MCTS", ERA3),
    ("v18", "Black-box pivot: ChangeNet+graph", ERA3),
    ("v19", "Synthesis: BFS-first + ExIt flywheel", ERA3),
]

# v13 search ladder rungs (v13_3 README + CHRONOS_EVOLUTION §4 v13).
LADDER = [
    "BFS sprint", "sense", "IW(1)", "IW(2)", "EHC",
    "waypoint", "A*", "greedy", "rescues",
]


# ==========================================================================
# FIGURE 0 — Cover image (required by the writeup)
# ==========================================================================

def fig_cover():
    fig, ax = plt.subplots(figsize=(12, 6.4))
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.4)

    ax.text(0.4, 5.55, "CHRONOS SOLVER", fontsize=34, fontweight="bold", color=INK)
    ax.text(
        0.42, 4.95,
        "BFS-First Genuine Search for ARC-AGI-3",
        fontsize=18, color=ERA2, fontweight="bold",
    )
    ax.text(
        0.42, 4.5,
        "An honest account of 19 iterations — and why live symbolic search beat the neural net",
        fontsize=12.5, color=MUTE, style="italic",
    )

    # headline numbers
    def stat(x, big, small, col):
        ax.text(x, 3.05, big, fontsize=40, fontweight="bold", color=col, ha="center")
        ax.text(x, 2.35, small, fontsize=11.5, color=MUTE, ha="center")

    stat(2.1, "0.22", "live white-box BFS\n(v12, ARC-AGI-3)", WIN)
    stat(6.0, "0.01", "black-box neural only\n(v18/v19 ablation)", LOSS)
    stat(9.9, "31 / 13", "levels / games\nsolved from scratch", ERA3)

    ax.annotate(
        "22× gap",
        xy=(4.05, 3.1), xytext=(4.05, 1.55),
        ha="center", fontsize=12, color=INK, fontweight="bold",
        arrowprops=dict(arrowstyle="-", color=MUTE, lw=1),
    )

    # three-era ribbon — three equal segments, short two-line labels
    y0, h = 0.78, 0.66
    x_lo, x_hi = 0.4, 11.6
    seg = (x_hi - x_lo) / 3.0
    ribbon = [("Era I · LLM swarm\nv1–v11", ERA1),
              ("Era II · Symbolic search\nv12–v13   ★ 0.22", ERA2),
              ("Era III · Model-based RL\nv14–v19", ERA3)]
    for i, (label, col) in enumerate(ribbon):
        x = x_lo + i * seg
        ax.add_patch(FancyBboxPatch((x + 0.06, y0), seg - 0.12, h,
                                    boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=col, ec="none", alpha=0.92))
        ax.text(x + seg / 2, y0 + h / 2, label, ha="center", va="center",
                color="white", fontsize=10, fontweight="bold", linespacing=1.25)

    ax.text(0.42, 0.25, "ARC Prize 2026 · Paper Track · github.com/shreyasmahimkar/arc-agi-3",
            fontsize=9, color=MUTE)
    _save(fig, "fig0_cover.png")


# ==========================================================================
# THUMBNAIL — exact 560x280 card image required by the Kaggle writeup form
# ==========================================================================

def fig_thumb():
    # 5.6 x 2.8 inches @ 100 dpi -> exactly 560 x 280 px (Kaggle card size)
    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.add_patch(plt.Rectangle((0, 0), 10, 5, fc="#0f1b2d", ec="none"))  # deep navy
    ax.text(0.45, 4.05, "CHRONOS SOLVER", fontsize=21, fontweight="bold", color="white")
    ax.text(0.47, 3.35, "BFS-First Genuine Search · ARC-AGI-3", fontsize=10.5, color="#9bc1e8")

    ax.text(1.7, 1.75, "0.22", fontsize=30, fontweight="bold", color="#5b9bd5", ha="center")
    ax.text(1.7, 0.85, "live white-box BFS", fontsize=8, color="#c7d3e0", ha="center")
    ax.text(5.0, 1.75, "vs", fontsize=14, color="#7c8aa0", ha="center")
    ax.text(8.0, 1.75, "0.01", fontsize=30, fontweight="bold", color="#e0726c", ha="center")
    ax.text(8.0, 0.85, "black-box neural", fontsize=8, color="#c7d3e0", ha="center")
    ax.text(5.0, 0.55, "22x gap  ·  19 iterations, one honest lesson", fontsize=8.2,
            color="#9aa7b8", ha="center")
    # save at 100 dpi for exact pixel size
    path = os.path.join(FIG_DIR, "thumbnail_560x280.png")
    fig.savefig(path, dpi=100, bbox_inches=None, pad_inches=0, facecolor="#0f1b2d")
    plt.close(fig)
    print("  wrote", os.path.relpath(path, REPO), "(560x280)")


# ==========================================================================
# FIGURE 1 — The headline: genuine search vs black-box
# ==========================================================================

def fig_scores():
    fig, ax = plt.subplots(figsize=(8.6, 5))
    labels = list(SCORES.keys())
    vals = [SCORES[k][0] for k in labels]
    cols = [SCORES[k][1] for k in labels]
    bars = ax.bar(labels, vals, color=cols, width=0.62, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.2f}",
                ha="center", va="bottom", fontweight="bold", fontsize=13)
    ax.set_ylabel("ARC-AGI-3 leaderboard score")
    ax.set_ylim(0, 0.26)
    ax.set_title("Genuine live search scores 22× the black-box neural agent",
                 fontsize=14, fontweight="bold", loc="left")
    ax.margins(x=0.04)
    _caption(fig, "The competition ships game sources in environment_files/, so live BFS reaches them and generalises to "
                  "the scored set. Black-box exploration alone is too weak; the v19 submission scored 0.02 because the "
                  "BFS plumbing failed to engage — a deployment bug, not a capability gap.")
    _save(fig, "fig1_scores.png")


# ==========================================================================
# FIGURE 2 — The three-era evolution timeline
# ==========================================================================

def fig_timeline():
    fig, ax = plt.subplots(figsize=(11.5, 8.2))
    ax.axis("off")
    n = len(TIMELINE)
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, n - 0.5)
    for i, (ver, desc, col) in enumerate(TIMELINE):
        y = n - 1 - i
        ax.add_patch(FancyBboxPatch((0.2, y - 0.32), 1.05, 0.64,
                                    boxstyle="round,pad=0.02,rounding_size=0.1",
                                    fc=col, ec="none"))
        ax.text(0.72, y, ver, ha="center", va="center", color="white",
                fontweight="bold", fontsize=11)
        ax.text(1.5, y, desc, ha="left", va="center", fontsize=11, color=INK)
        if ver == "v12":
            ax.text(9.8, y, "★ 0.22", ha="right", va="center", color=WIN,
                    fontweight="bold", fontsize=12)
    # era brackets
    eras = [("Era I\nLLM orchestration", ERA1, 0, 10),
            ("Era II\nsymbolic search", ERA2, 11, 12),
            ("Era III\nmodel-based RL", ERA3, 13, 18)]
    for label, col, lo, hi in eras:
        ya = n - 1 - hi - 0.42
        yb = n - 1 - lo + 0.42
        ax.add_patch(plt.Rectangle((0.05, ya), 0.08, yb - ya, fc=col, ec="none"))
        ax.text(-0.1, (ya + yb) / 2, label, ha="right", va="center",
                fontsize=9.5, color=col, fontweight="bold", rotation=0)
    ax.set_title("Three eras, each diagnosing the last era's bottleneck",
                 fontsize=14, fontweight="bold", loc="left", x=0.0)
    _caption(fig, "Era I proved LLM orchestration is too slow/unreliable for precise long-horizon action sequencing. "
                  "Era II reframed the task as search and scored. Era III pursues the generalisation ceiling with a "
                  "learned world model — while keeping BFS-first as the floor.")
    _save(fig, "fig2_timeline.png")


# ==========================================================================
# FIGURE 3 — v12 coverage (real data from caches)
# ==========================================================================

def fig_coverage():
    games = sorted(V12, key=lambda g: (-len(V12[g]), g))
    counts = [len(V12[g]) for g in games]
    total_lv = sum(counts)
    fig, ax = plt.subplots(figsize=(10, 5))
    cols = [ERA2 if c >= 4 else "#9bb8d8" for c in counts]
    bars = ax.bar(games, counts, color=cols, zorder=3, width=0.7)
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c + 0.08, str(c),
                ha="center", va="bottom", fontsize=10, color=INK)
    ax.set_ylabel("levels solved (genuine BFS)")
    ax.set_xlabel("ARC-AGI-3 game id")
    ax.set_ylim(0, max(counts) + 1)
    ax.set_title(f"v12 solved {total_lv} levels across {len(games)} games from scratch",
                 fontsize=14, fontweight="bold", loc="left")
    _caption(fig, "Source: the v12_bfs_cache_*.json solution files in this repo (each maps level -> action list). "
                  "These are paths the solver found live, not stored answers — every one was replay-verified on the real engine.")
    _save(fig, "fig3_coverage.png")


# ==========================================================================
# FIGURE 4 — The chaining fix (correctness + shorter solutions)
# ==========================================================================

def fig_chaining():
    levels = list(CHAIN.keys())
    synth = [CHAIN[k][0] for k in levels]
    chained = [CHAIN[k][1] for k in levels]
    x = np.arange(len(levels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.4, 5))
    b1 = ax.bar(x - w / 2, synth, w, label="set_level() synthetic baseline (wrong)",
                color=LOSS, zorder=3)
    b2 = ax.bar(x + w / 2, chained, w, label="chained L0..L(n-1) baseline (correct)",
                color=WIN, zorder=3)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5,
                    str(int(b.get_height())), ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x, levels)
    ax.set_ylabel("solution length (actions)")
    ax.set_ylim(0, max(synth) + 28)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    ax.set_title("Chaining real baselines fixed correctness — and halved solutions",
                 fontsize=13.5, fontweight="bold", loc="left")
    _caption(fig, "set_level(N)+RESET produces a different start state than naturally advancing from L(N-1) (player pos, "
                  "carried-key rotation, ~1400px frame diff on ls20). Plans solved from the synthetic baseline FAIL on replay. "
                  "Building level N from the chained L0..L(n-1) solutions fixed it — and shortened paths (e.g. L2 97->39).")
    _save(fig, "fig4_chaining.png")


# ==========================================================================
# FIGURE 5 — The ls20 difficulty curve and "the wall"
# ==========================================================================

def fig_wall():
    ls = V12.get("ls20", {0: 13, 1: 45, 2: 39, 3: 43, 4: 44})
    levels = sorted(ls)
    depths = [ls[i] for i in levels]
    fig, ax = plt.subplots(figsize=(8.6, 5))
    ax.plot([f"L{i}" for i in levels], depths, "-o", color=ERA2, lw=2.4,
            markersize=9, zorder=3, label="solved (genuine BFS depth)")
    for i, d in zip(levels, depths):
        ax.text(i, d + 1.4, str(d), ha="center", fontsize=10)
    # the wall at L5
    ax.scatter([len(levels)], [max(depths) + 6], s=140, marker="X", color=LOSS,
               zorder=4, label="L5 'the wall' — breadth-death (unsolved)")
    ax.axvspan(len(levels) - 0.5, len(levels) + 0.5, color=LOSS, alpha=0.06)
    ax.set_xlim(-0.5, len(levels) + 0.5)
    ax.set_xticks(range(len(levels) + 1))
    ax.set_xticklabels([f"L{i}" for i in levels] + ["L5"])
    ax.set_ylabel("search depth at solve (actions)")
    ax.set_title("ls20: deep levels die of breadth, not walls", fontsize=14,
                 fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    _caption(fig, "BFS solves L0–L4 (depth up to 44) but L5 (a dual-key puzzle, depth ~45+) exhausts the 500k-state budget. "
                  "The diagnosis — breadth-death, not an unsolvable wall — motivated Era III's learned heuristic / value prior.")
    _save(fig, "fig5_wall.png")


# ==========================================================================
# FIGURE 6 — The world-model memorisation fix (theory in one chart)
# ==========================================================================

def fig_wm_fix():
    metrics = list(WM_FIX.keys())
    v14 = [WM_FIX[m][0] for m in metrics]
    v15 = [WM_FIX[m][1] for m in metrics]
    x = np.arange(len(metrics))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.4, 5))
    b1 = ax.bar(x - w / 2, v14, w, label="v14 (pooled bottleneck → memorises)", color="#c9a3c4", zorder=3)
    b2 = ax.bar(x + w / 2, v15, w, label="v15 (cross-attend tokens → learns deltas)", color=ERA3, zorder=3)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5,
                    f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x, metrics)
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(0, 108)
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    ax.set_title("Make 'copy' the default, spend capacity on dynamics",
                 fontsize=14, fontweight="bold", loc="left")
    _caption(fig, "v14 predicted 64 next-frame tokens from a pooled belief with static queries — high train acc but it "
                  "MEMORISED trajectories (fresh-episode 36.6%). v15 cross-attends over the current frame's tokens, so "
                  "identity-copy is the residual default and capacity learns only the deltas → fresh-episode >90%.")
    _save(fig, "fig6_wm_fix.png")


# ==========================================================================
# FIGURE 7 — Honest negatives: the disproved colour-aug hypothesis
# ==========================================================================

def fig_honest():
    fig, ax = plt.subplots(figsize=(8.6, 5))
    labels = list(COLOUR_AUG.keys())
    vals = [COLOUR_AUG[k] for k in labels]
    bars = ax.barh(labels, vals, color=[LOSS, LOSS], zorder=3, height=0.5)
    ax.axvline(0, color=MUTE, lw=1.2)
    for b, v in zip(bars, vals):
        ax.text(v - 0.003, b.get_y() + b.get_height() / 2, f"{v:+.3f}",
                ha="right", va="center", fontsize=11, color=INK, fontweight="bold")
    ax.set_xlim(-0.08, 0.04)
    ax.set_xlabel("held-out changed-pixel accuracy lift from colour-permutation (+D4) augmentation")
    ax.set_title("A disproved hypothesis, reported as one",
                 fontsize=14, fontweight="bold", loc="left")
    ax.text(0.02, 1.5, f"world-model transfer\nplateaued at ~{WM_PLATEAU:.2f}",
            fontsize=10, color=MUTE, ha="left", va="center")
    _caption(fig, "Hypothesis: the WM over-fits colour, so colour-permutation augmentation should help. A/B test, logged "
                  "twice: lift = -0.02 (Mac) and -0.061 (RTX). The hypothesis was killed, not tuned. The negative points at a "
                  "representation/architecture limit (object-centric features), NOT a compute limit — so the next lever is not a bigger GPU.")
    _save(fig, "fig7_honest_negative.png")


# ==========================================================================
# FIGURE 8 — The v19 architecture: BFS-first routing cascade
# ==========================================================================

def _box(ax, xy, w, h, text, fc, tc="white", fs=10, bold=True):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fc, ec="none"))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
            color=tc, fontsize=fs, fontweight="bold" if bold else "normal")


def _arrow(ax, p1, p2, text="", col=MUTE):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=14,
                                 color=col, lw=1.6))
    if text:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx, my + 0.12, text, ha="center", fontsize=8.5, color=INK,
                fontweight="bold")


def fig_architecture():
    fig, ax = plt.subplots(figsize=(12, 6.6))
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.0)

    # --- top band: the per-game routing cascade (y 3.4 .. 6.2) ---
    _box(ax, (0.3, 4.55), 2.0, 0.9, "New game\n(frames only)", INK, fs=10)
    _box(ax, (2.85, 4.55), 2.3, 0.9, "White-box source\nreachable?", "#5b6172", fs=10)

    # YES -> BFS ladder (the 0.22 engine), top-right
    _box(ax, (6.0, 5.55), 3.2, 0.95, "BFS SEARCH LADDER\n(genuine live solve → 0.22)", ERA2, fs=10.5)
    # NO -> black-box, middle-right
    _box(ax, (6.0, 4.05), 3.2, 0.9, "Black-box ChangeNet\n+ exploration graph", ERA3, fs=10)
    # timeout backstop, lower-right
    _box(ax, (6.0, 2.7), 3.2, 0.85, "Solution cache\n(timeout backstop only)", "#9aa1b2", fs=9.5)

    _arrow(ax, (2.3, 5.0), (2.85, 5.0))
    _arrow(ax, (5.15, 5.2), (6.0, 6.0), "source found", WIN)
    _arrow(ax, (5.15, 4.8), (6.0, 4.5), "no source", ERA3)
    _arrow(ax, (7.6, 5.55), (7.6, 4.95), "times out", MUTE)
    _arrow(ax, (7.6, 4.05), (7.6, 3.55), "times out", MUTE)

    # ladder rungs strip (far right, within bounds)
    ax.text(9.4, 6.45, "ladder rungs:", fontsize=8.5, color=MUTE, ha="left")
    ax.text(9.4, 6.12, " → ".join(LADDER[:4]), fontsize=7.6, color=ERA2, ha="left")
    ax.text(9.4, 5.86, " → ".join(LADDER[4:]), fontsize=7.6, color=ERA2, ha="left")

    # --- divider ---
    ax.plot([0.3, 11.7], [2.15, 2.15], color=GRID, lw=1.4)

    # --- bottom band: the ExIt flywheel as a left-to-right loop (y < 2.0) ---
    ax.text(0.3, 1.75, "ExIt flywheel  (offline research engine — lifts the learned prior)",
            fontsize=10, color=ERA3, fontweight="bold", ha="left")
    steps = ["solve\n(BFS expert)", "harvest\n(real engine)", "train WM\n(dynamics)", "plan (MPC)\nin imagination"]
    bx, bw, by, bh, gap = 0.3, 2.45, 0.45, 0.85, 0.45
    centers = []
    for i, s in enumerate(steps):
        x = bx + i * (bw + gap)
        _box(ax, (x, by), bw, bh, s, ERA3, fs=8.8)
        centers.append((x, x + bw))
        if i > 0:
            _arrow(ax, (centers[i - 1][1], by + bh / 2), (x, by + bh / 2), col=ERA3)
    # return loop arrow
    last_r = centers[-1][1]
    ax.add_patch(FancyArrowPatch((last_r, by + bh + 0.05), (centers[0][0], by + bh + 0.05),
                                 connectionstyle="arc3,rad=-0.25", arrowstyle="-|>",
                                 mutation_scale=14, color=ERA3, lw=1.6))
    ax.text((centers[0][0] + last_r) / 2, by + bh + 0.62, "verify on real engine → retrain (AlphaZero-style)",
            ha="center", fontsize=8.2, color=ERA3)

    ax.set_title("v19 routing: BFS-first, learned fallback, cache only on timeout",
                 fontsize=14, fontweight="bold", loc="left", x=0.02, y=0.98)
    _caption(fig, "Genuine search always goes first. The black-box agent (fusing the two ARC-AGI-3 preview winners — "
                  "StochasticGoose's ChangeNet + Blind Squirrel's transition graph) covers hidden/source-unreachable games. "
                  "The cache is a timeout backstop, never the first move. The ExIt flywheel (solve→harvest→train→plan) is the "
                  "offline engine for lifting the learned prior.")
    _save(fig, "fig8_architecture.png")


# ==========================================================================
if __name__ == "__main__":
    print("Generating figures into", os.path.relpath(FIG_DIR, REPO))
    fig_thumb()
    fig_cover()
    fig_scores()
    fig_timeline()
    fig_coverage()
    fig_chaining()
    fig_wall()
    fig_wm_fix()
    fig_honest()
    fig_architecture()
    print("Done. %d figures." % len(glob.glob(os.path.join(FIG_DIR, "*.png"))))
