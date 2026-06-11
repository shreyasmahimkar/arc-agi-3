# fable5 — pure ARC-AGI-3 run with claude-fable-5

Plays an ARC-AGI-3 game (default: **LS20**, locksmith) on the official
`three.arcprize.org` API with an officially tracked scorecard, using
`claude-fable-5` as the agent brain. No framework — raw REST + Anthropic SDK.

## Setup (with venv)

```bash
cd ~/gitrepos/OpenSource/kaggle/arc3/fable5

# create + activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# install dependencies
pip install requests anthropic python-dotenv

# create .env if it doesn't exist yet, then add your keys
[ -f .env ] || cp .env.example .env   # ARC_API_KEY + ANTHROPIC_API_KEY
```

## Run

```bash
source .venv/bin/activate   # if not already active
python fable5_agent.py --game ls20 --max-actions 250
```

Start with `--max-actions 50` for a cheap sanity check before a full run.

## How it works

1. Opens a scorecard (`POST /api/scorecard/open`, tagged `fable5`).
2. `RESET` starts LS20; every turn the 64x64 frame is rendered as a hex grid
   and sent to claude-fable-5 along with a cell-level diff of the last action,
   the model's own persistent "memory" notes, and available actions.
3. The model replies with JSON: observation, updated memory, and the next
   action (`ACTION1-7`, `ACTION6` with x,y click coords). Reasoning is attached
   to each action so it shows up in the replay.
4. `GAME_OVER` auto-resets; `WIN` stops the run.
5. Closes the scorecard and prints the official result + replay URL:
   `https://three.arcprize.org/scorecards/<card_id>`

Each run also saves `run_ls20_<timestamp>.json` with every step, the final
scorecard, and token usage.

## Knobs

| Flag | Default | |
|---|---|---|
| `--game` | `ls20` | any game prefix from `/api/games` |
| `--max-actions` | `250` | API budget for the run |
| `--model` | `claude-fable-5` | any Anthropic model string |
| `--history-turns` | `6` | conversation turns kept in context |
| `--render` | off | draw the colored game grid live in the terminal |

## Watching it play

- Live: `python fable5_agent.py --game ls20 --max-actions 50 --render`
- Replay in browser after the run: `https://three.arcprize.org/scorecards/<card_id>`
- Play LS20 yourself: https://three.arcprize.org/games/ls20
