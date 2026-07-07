#!/bin/bash
# Build the Kaggle dataset for the v19 BFS-first submission (v19-to-kaggle.ipynb).
# Output is a FLAT folder: combined_agent.py + forge_agent.py + pretrained_weights.pt
# + solutions/ (timeout backstop). Upload it as a PRIVATE Kaggle dataset.
#
#   bash build_kaggle_dataset.sh                 # -> ~/Downloads/v19-forge
#   bash build_kaggle_dataset.sh /path/to/out
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/src"
ARCHIVE="$HERE/../archive"
OUT="${1:-$HOME/Downloads/v19-forge}"

rm -rf "$OUT"; mkdir -p "$OUT"
cp "$SRC/combined_agent.py" "$SRC/forge_agent.py" "$OUT/"           # entry + fallback
cp "$SRC/pretrained_weights.pt" "$OUT/" 2>/dev/null || echo "WARN: no pretrained_weights.pt (black-box prior will be COLD)"
cp -r "$SRC/solutions" "$OUT/solutions"                            # timeout backstop cache

# Enrich the ls20 backstop with the VERIFIED v13 L0-L4 (optimal counts 13/45/39/43/44,
# confirmed to replay-verify on the live version ls20-9607627b). Backstop only fires
# when live BFS times out on a level we've seen before.
if [ -f "$ARCHIVE/v13/v13_bfs_cache_ls20.json" ]; then
  cp "$ARCHIVE/v13/v13_bfs_cache_ls20.json" "$OUT/solutions/ls20.json"
fi

echo "----------------------------------------------------------------"
echo "Built Kaggle dataset at: $OUT"
ls -1 "$OUT"
echo "solutions/: $(ls "$OUT"/solutions/*.json | wc -l | tr -d ' ') games | ls20 backstop levels: $(python3 -c "import json;print(sorted(json.load(open('$OUT/solutions/ls20.json'))))" 2>/dev/null)"
echo "----------------------------------------------------------------"
echo "Next: upload this FOLDER as a private Kaggle dataset named 'v19-forge'."
