import arc_agi
arc = arc_agi.Arcade()
env = arc.make("ls20")
frame = env.reset()
print(dir(frame))
for key in dir(frame):
    if not key.startswith('_'):
        try:
            val = getattr(frame, key)
            if not callable(val):
                print(f"{key}: type={type(val)}")
        except Exception as e:
            pass
