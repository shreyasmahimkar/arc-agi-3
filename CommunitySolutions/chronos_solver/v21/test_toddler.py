#!/usr/bin/env python3
# Offline tests for the neural toddler's PURE parts (data IO, encoding, device
# pick, torch-absent fallback). Actual training runs on the Mac GPU (MPS).
#   ../../../.venv312/bin/python test_toddler.py            # offline checks
#   ../../../.venv312/bin/python test_toddler.py --train ls20   # train on Mac GPU
import os, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ["V21_TODDLER_DIR"] = tempfile.mkdtemp(prefix="v21tod_")

from brain import toddler_net as T


def _check(name, cond):
    print(("PASS " if cond else "FAIL ") + name); return cond


def offline():
    ok = True
    # data harvest round-trips
    T.append_samples("ls20", [
        {"frame": [[0, 1], [1, 0]], "action": 2, "changed": True, "won": False},
        {"frame": [[0, 0], [0, 0]], "action": 4, "changed": False, "won": False},
        {"frame": [[2, 2], [2, 2]], "action": 2, "changed": True, "won": True},
    ])
    rows = T.load_samples("ls20")
    ok &= _check("harvest append/load round-trips", len(rows) == 3 and rows[-1]["won"] is True)

    # encoding is fixed-size + clamped
    import numpy as np
    enc = T._encode([[99, 1], [1, 0]])   # 99 clamped to < N_COLORS
    ok &= _check("encode -> GRIDxGRID, clamped", enc.shape == (T.GRID, T.GRID) and enc.max() < T.N_COLORS)

    # device pick never raises + is a known string
    ok &= _check("device pick returns cpu/mps/cuda", T.pick_device() in ("cpu", "mps", "cuda"))

    # fallback: with no trained weights, order_actions returns the fallback order
    tod = T.ToddlerNet("ls20", fallback_order=[2, 1, 6, 3, 4, 5, 7])
    ok &= _check("order_actions falls back cleanly (no weights)",
                 tod.order_actions(frame=[[0, 1], [1, 0]]) == [2, 1, 6, 3, 4, 5, 7])
    ok &= _check("order_actions(no frame) -> fallback", tod.order_actions() == [2, 1, 6, 3, 4, 5, 7])

    print("torch available on this machine:", T.torch_available(), "| device:", T.pick_device())
    print("\n" + ("ALL TODDLER OFFLINE TESTS PASSED" if ok else "TODDLER TESTS FAILED"))
    return 0 if ok else 1


def train(game):
    tod = T.ToddlerNet(game)
    print(f"device = {tod.device} | torch = {T.torch_available()}")
    print(tod.train())
    print("sample order_actions:", tod.order_actions(frame=[[0, 1, 0], [1, 0, 1], [0, 1, 0]]))
    return 0


def main():
    if "--train" in sys.argv:
        g = sys.argv[sys.argv.index("--train") + 1] if len(sys.argv) > sys.argv.index("--train") + 1 else "ls20"
        return train(g)
    return offline()


if __name__ == "__main__":
    sys.exit(main())
