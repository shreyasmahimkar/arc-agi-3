cd /Users/shreyas/gitrepos/OpenSource/kaggle/arc3/CommunitySolutions/chronos_solver/v19
source /Users/shreyas/gitrepos/OpenSource/kaggle/arc3/.venv312/bin/activate

python play_game.py --game ls20          # opens the matplotlib render window, agent plays live
python play_game.py --game ar25          # any game with a shipped source
python play_game.py --game ls20 --fast   # headless (no window/PNGs), fast