import arc_agi
arc = arc_agi.Arcade()
env = arc.make("ls20")

# I don't know the exact actions to win level 0 instantly.
# But I can print the documentation or inspect env methods.
print("env methods:", [m for m in dir(env) if not m.startswith('_')])
