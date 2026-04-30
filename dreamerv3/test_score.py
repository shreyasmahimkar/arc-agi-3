import arc_agi
arc = arc_agi.Arcade()
env = arc.make("ls20")
print(dir(arc))
if hasattr(arc, "scorecard_manager"):
    sm = arc.scorecard_manager
    print(dir(sm))
