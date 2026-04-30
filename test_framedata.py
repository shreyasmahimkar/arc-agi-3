import arc_agi
arc = arc_agi.Arcade()
env = arc.make('ls20')
state = env.reset()
print(type(state))
print(dir(state))
try:
    print(state.model_dump())
except Exception as e:
    print("model_dump failed", e)
try:
    print(state.to_dict())
except Exception as e:
    print("to_dict failed", e)
