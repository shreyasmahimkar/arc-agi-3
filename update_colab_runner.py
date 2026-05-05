import json

with open('CommunitySolutions/Ash_solver/v3/colab_runner.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    source = cell.get('source', [])
    new_source = []
    for line in source:
        new_line = line.replace('CommunitySolutions/Ash_solver/requirements_colab.txt', 'CommunitySolutions/Ash_solver/v3/requirements_colab.txt')
        new_line = new_line.replace('CommunitySolutions/Ash_solver/play_game.py', 'CommunitySolutions/Ash_solver/v3/play_game.py')
        new_source.append(new_line)
    cell['source'] = new_source

with open('CommunitySolutions/Ash_solver/v3/colab_runner.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
