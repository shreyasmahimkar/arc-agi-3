import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

def read_file(filepath):
    with open(filepath, 'r') as f:
        return f.read()

# Read the local files
symlog_code = read_file("models/dreamer/symlog.py")
rssm_code = read_file("models/dreamer/rssm.py")
actor_critic_code = read_file("models/dreamer/actor_critic.py")
planner_code = read_file("models/dreamer/planner.py")
env_code = read_file("ls20_dreamer_env.py")

# Dynamically inject the Kaggle offline Arcade constructor for the notebook
env_code = env_code.replace(
    "self.arc = arc_agi.Arcade()",
    """import os
        import arc_agi
        self.arc = arc_agi.Arcade(
            environments_dir="/kaggle/input/competitions/arc-prize-2026-arc-agi-3/environment_files",
            operation_mode=arc_agi.OperationMode.OFFLINE
        )"""
)

play_code = read_file("play_and_learn_kaggle.py")

nb['cells'] = [
    nbf.v4.new_markdown_cell("# DreamerV3 + Neural Maps on Kaggle\n* Full Reinforcement Learning implementation from scratch.\n* Runs on NVIDIA GPUs natively."),
    
    nbf.v4.new_code_cell("!pip install --no-index --find-links /kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels arc-agi python-dotenv\n!pip install torch torchvision torchaudio"),
    
    nbf.v4.new_code_cell("!mkdir -p models/dreamer"),
    
    nbf.v4.new_code_cell(f"%%writefile models/dreamer/__init__.py\n# Init file"),
    
    nbf.v4.new_code_cell(f"%%writefile models/dreamer/symlog.py\n{symlog_code}"),
    
    nbf.v4.new_code_cell(f"%%writefile models/dreamer/rssm.py\n{rssm_code}"),
    
    nbf.v4.new_code_cell(f"%%writefile models/dreamer/actor_critic.py\n{actor_critic_code}"),
    
    nbf.v4.new_code_cell(f"%%writefile models/dreamer/planner.py\n{planner_code}"),
    
    nbf.v4.new_code_cell(f"%%writefile ls20_dreamer_env.py\n{env_code}"),
    
    nbf.v4.new_code_cell(f"%%writefile play_and_learn.py\n{play_code}"),
    
    nbf.v4.new_code_cell("""import os

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    # Wait for the ARC gateway to initialize
    !curl --fail --retry 99 --retry-all-errors --retry-delay 5 --retry-max-time 600 http://gateway:8001/api/games
    
    # Configure arc-agi to point to the local Kaggle gateway
    with open('.env', 'w') as f:
        f.write(\"\"\"SCHEME=http
HOST=gateway
PORT=8001
ARC_API_KEY=test-key-123
ARC_BASE_URL=http://gateway:8001/
OPERATION_MODE=online
RECORDINGS_DIR=/kaggle/working/server_recording
\"\"\")

    # Run the Dreamer Agent
    !MPLBACKEND=agg python play_and_learn.py
else:
    # Fast fallback for saving the notebook version
    import pandas as pd
    submission = pd.DataFrame(data=[['1_0', '1', True, 1]], columns=['row_id', 'game_id', 'end_of_game', 'score'])
    submission.to_parquet('/kaggle/working/submission.parquet', index=False)
""")
]

with open('dreamerv3-arc-submission.ipynb', 'w') as f:
    nbf.write(nb, f)
print('DreamerV3 Notebook created successfully!')
