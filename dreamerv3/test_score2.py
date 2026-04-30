import arc_agi
arc = arc_agi.Arcade()
env = arc.make("ls20")
print(dir(arc))
scorecard = arc.get_scorecard()
print(scorecard)
print(type(scorecard))
