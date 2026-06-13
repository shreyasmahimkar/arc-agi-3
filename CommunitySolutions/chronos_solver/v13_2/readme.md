cd CommunitySolutions/chronos_solver/v13_2

python benchmark.py \
    --games ls20:7,ar25:3 \
    --versions .,../v13_1 \
    --budget 600 --workers 8 --max-states 5000000