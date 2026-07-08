#!/usr/bin/env python3
# =====================================================================
# Standalone Mac test for the RUNTIME CODER (local Qwen via Ollama).
# Proves the on-the-fly world-model writer works end-to-end, isolated from
# the hour-long cadence: (1) Ollama reachable + model responds, (2) shows the
# actual WorldModel CODE the LLM writes, (3) makes it solve a real level on the
# live engine with BFS BYPASSED, and verifies the win.
#
# Run on the Mac (from the v21/ dir):
#   export V21_LLM_BACKEND=ollama V21_OLLAMA_MODEL=qwen2.5-coder:3b
#   ../../../.venv312/bin/python test_runtime_coder_mac.py            # default: ft09 L0
#   ../../../.venv312/bin/python test_runtime_coder_mac.py vc33 0     # game + level
#   ../../../.venv312/bin/python test_runtime_coder_mac.py ls20 5     # try a real wall
# =====================================================================
import os, sys, time, logging

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("V21_LLM_BACKEND", "ollama")
os.environ.setdefault("V21_OLLAMA_MODEL", "qwen2.5-coder:3b")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")      # CPU BFS; safe fork pools

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

import cadence_runner as cr
for p in (cr.V19_SRC, cr.V20_SRC):                     # v19/v20 on path (cr does this only in main)
    if p and p not in sys.path:
        sys.path.insert(0, p)
import llm_backend
import runtime_coder as rc


def main():
    gid = sys.argv[1] if len(sys.argv) > 1 else "ft09"
    lvl = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    max_len = int(os.environ.get("V21_RUNTIME_MAXLEN", "200"))
    print(f"\n=== runtime-coder test: {gid} L{lvl} (max_len={max_len}) ===")

    # [1] Ollama up + model actually responds -----------------------------------
    llm = llm_backend.get_backend()
    model = getattr(llm, "model", getattr(llm, "model_id", "?"))
    print(f"\n[1] LLM backend = {llm.name}  model={model}")
    if llm.name == "mock":
        print("    ⚠️  MOCK backend — Ollama not reachable or model not pulled.")
        print("    Fix:  ollama serve &   ;   ollama pull $V21_OLLAMA_MODEL   ; rerun")
        return 2
    t = time.time()
    try:
        out = llm.complete("Reply with only the word: ready", max_tokens=8)
        print(f"    model replied in {time.time()-t:.1f}s -> {out.strip()[:60]!r}")
    except Exception as e:
        print(f"    ❌ model call failed: {e}")
        print("    If HTTP 500: the model likely OOM'd — use a smaller one (qwen2.5-coder:3b).")
        return 1

    # [2] Show the actual WorldModel CODE the LLM writes -------------------------
    print(f"\n[2] Model writes a WorldModel for {gid} L{lvl}:")
    try:
        code = llm.complete(
            rc.WM_PROMPT.format(obs=f"game={gid} level={lvl}; frame is a 64x64 int color grid; "
                                    f"actions 1-5 discrete, 6=click(x,y), 7=undo"),
            system=rc.WM_SYSTEM, max_tokens=800)
        print("    ---- generated code (first 35 lines) ----")
        for line in code.splitlines()[:35]:
            print("    " + line)
        print("    -----------------------------------------")
        wm, err = rc._exec_world_model(code, {"game": gid, "level": lvl})
        print(f"    sandbox-exec: {'OK, WorldModel built' if wm else 'FAILED: '+str(err)}")
    except Exception as e:
        print(f"    (code display skipped: {e})")

    # [3] Solve the level with the runtime coder ONLY (BFS bypassed) -------------
    info = cr.resolve_source(gid)
    if not info:
        print(f"\n[3] no engine source for {gid} — cannot run the live solve"); return 1
    path, cls, ver = info
    from combined_agent import BFSSolver
    solver = BFSSolver(path, cls, bfs_timeout=60)
    if not solver.load():
        print("[3] solver.load() failed"); return 1
    print(f"\n[3] Solving {gid} L{lvl} via runtime coder ONLY (engine={ver}, BFS bypassed)…")
    t = time.time()
    plan = cr._runtime_coder_for_solver(solver, lvl, llm, max_len)
    dt = time.time() - t
    if plan and solver.verify_solution(lvl, plan):
        print(f"\n✅ SOLVED by generated code: {len(plan)} actions in {dt:.1f}s")
        print(f"   plan (first 12): {plan[:12]}")
        return 0
    print(f"\n❌ runtime coder did not solve {gid} L{lvl} in {dt:.1f}s "
          f"(plan={'none' if not plan else str(len(plan))+' actions, failed verify'})")
    print("   Try an easier level (ft09 0 / vc33 0) or raise V21_RUNTIME_MAXLEN.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
