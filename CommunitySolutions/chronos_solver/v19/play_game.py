"""Play an ARC-AGI-3 game with the v19 agent (combined_agent.MyAgent), rendered.

Watch the BFS-first v19 agent solve a game live in a window:

  python play_game.py                      # ls20, matplotlib 'human' render
  python play_game.py --game ar25
  python play_game.py --game ls20 --fast   # no window/PNGs (headless / fast iterate)
  python play_game.py --game ls20 --max-steps 300

It drives the REAL v19 agent (live white-box BFS first; black-box ForgeAgent only
if no source; cached solution only on BFS timeout when V19_CACHE_FALLBACK=1). The
env's local_dir is pointed at environment_files/<game> so the agent's BFS finds the
shipped source — exactly the path that scored 0.22 on Kaggle.
"""
import time, sys, os, glob, logging

# optional .env (offline play needs none, but mirror v13_3)
_envp = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_envp):
    for line in open(_envp):
        if "=" in line and not line.startswith("#"):
            k, v = line.strip().split("=", 1); os.environ[k] = v.strip("\"'")

import matplotlib
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(repo_root, "arc_solver"))
sys.path.insert(0, os.path.join(repo_root, "arc-prize-2026-arc-agi-3", "ARC-AGI-3-Agents"))
sys.path.insert(0, os.path.dirname(__file__))          # so `import combined_agent` wins

import arc_agi
from arcengine import GameAction, GameState, ActionInput
from combined_agent import MyAgent           # <-- the v19 agent (BFS-first + fallbacks)
import combined_agent

log_file = os.path.join(os.path.dirname(__file__), "v19_play.log")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

ARC_COLORS = ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00', '#AAAAAA',
              '#F012BE', '#FF851B', '#7FDBFF', '#870C25']
cmap = ListedColormap(ARC_COLORS)


def save_frame_as_image(frame_data, current_level, step_count, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    if not getattr(frame_data, "frame", None):
        return
    plt.figure(figsize=(4, 4))
    plt.imshow(frame_data.frame[0], cmap=cmap, vmin=0, vmax=9)
    plt.xticks([]); plt.yticks([])
    plt.savefig(os.path.join(output_dir, f"level_{current_level:02d}_step_{step_count:04d}.png"),
                bbox_inches="tight", pad_inches=0)
    plt.close()


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Play an ARC-AGI-3 game with the v19 agent")
    ap.add_argument("--game", type=str, default="ls20")
    ap.add_argument("--fast", action="store_true", help="no render window / PNGs (headless)")
    ap.add_argument("--max-steps", type=int, default=300, help="max steps per level")
    args = ap.parse_args()
    if args.fast:
        matplotlib.use("Agg")

    GAME = args.game
    env_dir = os.path.join(repo_root, "arc-prize-2026-arc-agi-3", "environment_files")
    logger.info("=" * 46)
    logger.info(f"=== v19 agent (combined_agent.MyAgent) — play {GAME} ===")
    logger.info("=" * 46)

    arc = arc_agi.Arcade(environments_dir=env_dir, operation_mode=arc_agi.OperationMode.OFFLINE)
    env = arc.make(GAME, render_mode=None if args.fast else "human")

    # CRITICAL: the env already knows the EXACT game version it loaded (multiple
    # versions can live under environment_files/<game>/<hash>/). Use the env's own
    # local_dir — NEVER override it with a bare glob, or the BFS solves a DIFFERENT
    # version than the env runs and its plans fail at the first level whose layout
    # differs (this was the ls20-L1 "failure"). Only fall back to glob if missing.
    class DummyEnvInfo:
        def __init__(self, d): self.local_dir = d
    ei = getattr(env, "environment_info", None)
    if not ei or not getattr(ei, "local_dir", None):
        matches = glob.glob(os.path.join(env_dir, GAME, "**", f"{GAME}.py"), recursive=True)
        fallback = os.path.dirname(matches[0]) if matches else os.path.join(env_dir, GAME)
        env.environment_info = DummyEnvInfo(fallback)
    game_local_dir = env.environment_info.local_dir

    logger.info(f"[1] MyAgent for {GAME} (source dir {game_local_dir})")
    agent = MyAgent(card_id="", game_id=GAME, agent_name="v19", ROOT_URL="",
                    record=False, arc_env=env)

    # v13_3 carried a deepcopy/camera pickling fix; combined_agent's BFS loads its
    # own game from disk, so it's only patched if the hook actually exists.
    if hasattr(combined_agent, "_fast_deepcopy"):
        import copy
        def stable_deepcopy(game):
            cam = getattr(game, "_camera", None)
            if cam is not None: game._camera = None
            g = copy.deepcopy(game)
            if cam is not None: game._camera = cam; g._camera = cam
            return g
        combined_agent._fast_deepcopy = stable_deepcopy

    logger.info("[2] reset + play loop")
    images_dir = os.path.join(os.path.dirname(__file__), "images", GAME)
    reset_out = env.reset()
    lf = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    frames = [lf]; agent.append_frame(lf)
    if not args.fast: save_frame_as_image(lf, getattr(lf, "levels_completed", 0), 0, images_dir)

    step = level_step = 0
    current_level = getattr(lf, "levels_completed", 0)
    while True:
        step += 1; level_step += 1
        if level_step > args.max_steps:
            logger.info(f"max steps ({args.max_steps}) on level {current_level} — stop"); break
        if agent.is_done(frames, lf) or getattr(lf, "state", None) in (GameState.GAME_OVER, GameState.WIN):
            logger.info(f"game finished — state={getattr(lf,'state','?')}"); break

        t0 = time.time()
        try:
            action = agent.choose_action(frames, lf)
        except Exception as e:
            logger.info(f"choose_action failed: {e}"); break
        dt = time.time() - t0

        act_id = getattr(action, "value", None)
        if act_id is None and hasattr(action, "id"):
            act_id = getattr(action.id, "value", None)
        act_name = getattr(getattr(action, "id", action), "name", f"ACTION_{act_id}")
        reasoning = getattr(action, "reasoning", "")
        logger.info(f"--- step {step} | {act_name} (id={act_id}) {dt:.2f}s | {reasoning}")

        try:
            if act_id == 6:
                x = y = 0
                if hasattr(action, "data") and isinstance(action.data, dict):
                    x, y = action.data.get("x", 0), action.data.get("y", 0)
                step_result = env.step(GameAction.ACTION6, data={"x": int(x), "y": int(y)})
            elif isinstance(action, GameAction):
                step_result = env.step(action)
            else:
                step_result = env.step(GameAction.from_id(act_id))
        except Exception as e:
            logger.info(f"env.step failed: {e}"); break

        if isinstance(step_result, tuple):
            lf = step_result[0]
        else:
            lf = step_result
        frames.append(lf); agent.append_frame(lf)

        new_level = getattr(lf, "levels_completed", 0)
        if new_level != current_level:
            logger.info(f"*** advanced to level {new_level} ***")
            current_level = new_level; level_step = 0
        logger.info(f"    level={current_level} state={getattr(lf,'state','?')} lvl_steps={level_step}")
        if not args.fast:
            save_frame_as_image(lf, current_level, step, images_dir); time.sleep(0.02)

    logger.info("=" * 46)
    try:
        logger.info(f"SCORECARD (RHAE):\n{arc.get_scorecard()}\nTOTAL STEPS: {step}")
    except Exception as e:
        logger.info(f"scorecard unavailable: {e}")


if __name__ == "__main__":
    main()
