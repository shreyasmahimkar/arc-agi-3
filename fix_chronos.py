import json

# 1. Update colab_runner.ipynb
with open('CommunitySolutions/chronos_solver/colab_runner.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    new_source = []
    for line in cell.get('source', []):
        new_line = line.replace('CommunitySolutions/Ash_solver/v3/requirements_colab.txt', 'CommunitySolutions/chronos_solver/requirements_colab.txt')
        new_line = new_line.replace('CommunitySolutions/Ash_solver/v3/play_game.py', 'CommunitySolutions/chronos_solver/play_game.py')
        new_source.append(new_line)
    cell['source'] = new_source

with open('CommunitySolutions/chronos_solver/colab_runner.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

# 2. Update Kaggle notebook
with open('CommunitySolutions/Ash_solver/v3/shreyas-m-arc-agi-agent-v3.ipynb', 'r') as f:
    knb = json.load(f)

with open('CommunitySolutions/chronos_solver/my_agent.py', 'r') as f:
    agent_code = f.read()

for cell in knb.get('cells', []):
    if cell.get('cell_type') == 'code' and any('%%writefile' in line and 'my_agent.py' in line for line in cell.get('source', [])):
        lines = ['%%writefile /kaggle/working/my_agent.py\n'] + [line + '\n' for line in agent_code.split('\n')]
        lines[-1] = lines[-1].rstrip('\n')
        cell['source'] = lines
        break

with open('CommunitySolutions/chronos_solver/shreyas-m-arc-agi-agent-chronos.ipynb', 'w') as f:
    json.dump(knb, f, indent=1)
