# v13 vs v13_1 benchmark

Fresh solvers, no caches/frontiers. 600s search budget per level per version, workers=8, max-states=5000000. v13 = bfs then greedy (budget split); v13_1 = auto ladder (bfs->waypoint->astar->iw1->iw2->greedy->rescues).

| Game | Lvl | v13_2 solved | v13_2 time | v13_2 acts | v13_1 solved | v13_1 time | v13_1 acts | winning rung |
|---|---|---|---|---|---|---|---|---|
| ar25 | 0 | YES | 12.1s | 15 | YES | 12.3s | 15 | iw1 |
| ar25 | 1 | YES | 28.6s | 11 | YES | 27.4s | 11 | iw2 |
| ar25 | 2 | no | 603.5s | - | no | 603.5s | - |  |
| ls20 | 0 | YES | 5.0s | 13 | YES | 3.0s | 13 | bfs |
| ls20 | 1 | YES | 107.2s | 45 | YES | 50.4s | 45 | astar |
| ls20 | 2 | YES | 72.7s | 39 | YES | 25.6s | 39 | waypoint |
| ls20 | 3 | YES | 133.7s | 43 | YES | 102.6s | 43 | bfs |
| ls20 | 4 | YES | 379.9s | 44 | YES | 347.2s | 44 | bfs |
| ls20 | 5 | no | 600.7s | - | no | 600.7s | - |  |
| ls20 | 6 | no | 600.5s | - | no | 600.4s | - |  |

## Totals

- **v13_2**: 7/10 levels solved, 2544s total search time
- **v13_1**: 7/10 levels solved, 2373s total search time
