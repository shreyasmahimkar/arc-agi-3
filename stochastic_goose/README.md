# ARC3 Local Environment Guide

This guide provides step-by-step instructions on how to run Python scripts in this environment using the locally configured virtual environment.

## Prerequisites

Ensure you are using the terminal and are currently in the project root directory:
```bash
cd /Users/shreyas/gitrepos/OpenSource/kaggle/arc3
```

## Step-by-Step Instructions

### 1. Activate the Virtual Environment
This project uses a specific Python 3.12 virtual environment named `.venv312`. To run scripts correctly with all the installed dependencies (like the patched ARC-AGI wheels), you must activate this environment first.

Run the following command in your terminal:
```bash
source .venv312/bin/activate
```
*(You should see `(.venv312)` appear at the beginning of your terminal prompt, indicating the environment is active.)*

### 2. Run a Python File
Once the environment is active, you can run any Python file using the `python` command. 

For example, to run the manual play script:
```bash
python manual_play.py
```

Or to run the quickstart script:
```bash
python run_quickstart.py
```

### 3. Deactivate the Virtual Environment (Optional)
When you are done working in this project and want to return to your global Python environment, you can deactivate the virtual environment by simply running:
```bash
deactivate
```
