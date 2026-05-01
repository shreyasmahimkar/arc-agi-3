import arc_agi
arc = arc_agi.Arcade()
env = arc.make('ls20', render_mode='human')
obs = env.reset()
print("frame:", obs.frame)
