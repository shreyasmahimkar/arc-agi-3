import arc_agi
from arcengine import GameAction
arc = arc_agi.Arcade()
env = arc.make("ls20")
frame = env.reset()
steps = 0
resets = 0
while True:
    steps += 1
    # ACTION1 is usually UP or something.
    frame = env.step(GameAction.ACTION1)
    if frame is None:
        break
    if frame.full_reset:
        print(f"Step {steps}: full_reset triggered")
    if frame.state.name == "GAME_OVER":
        resets += 1
        print(f"Step {steps}: GAME_OVER triggered (death #{resets})")
        if resets >= 4:
            break
