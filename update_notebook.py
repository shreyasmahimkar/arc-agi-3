import json

# Read new agent code
with open('CommunitySolutions/Ash_solver/v3/my_agent.py', 'r') as f:
    agent_code = f.read()

# Read notebook
with open('CommunitySolutions/Ash_solver/shreyas-m-arc-agi-agent-v2.ipynb', 'r') as f:
    nb = json.load(f)

# Find cell that writes my_agent.py
for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code' and any('%%writefile' in line and 'my_agent.py' in line for line in cell.get('source', [])):
        # Replace source
        lines = ['%%writefile /kaggle/working/my_agent.py\n'] + [line + '\n' for line in agent_code.split('\n')]
        # Remove trailing newline from last element
        lines[-1] = lines[-1].rstrip('\n')
        cell['source'] = lines
        break

# Save updated notebook
with open('CommunitySolutions/Ash_solver/v3/shreyas-m-arc-agi-agent-v3.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
