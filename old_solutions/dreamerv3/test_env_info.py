import arc_agi
from arcengine import GameAction
arc = arc_agi.Arcade()
env = arc.make("ls20")
frame = env.reset()
print(f"Grid size: {len(frame.frame)} x {len(frame.frame[0])}")
print("Frame attributes:")
for attr in dir(frame):
    if not attr.startswith('_') and not callable(getattr(frame, attr)):
        val = getattr(frame, attr)
        # truncate large values
        if isinstance(val, (list, tuple)) and len(val) > 5:
            print(f"  {attr}: list/tuple of length {len(val)}")
        else:
            print(f"  {attr}: {val}")

