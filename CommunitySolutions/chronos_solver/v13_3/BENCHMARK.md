# v13 vs v13_1 benchmark

Fresh solvers, no caches/frontiers. 600s search budget per level per version, workers=8, max-states=5000000. v13 = bfs then greedy (budget split); v13_1 = auto ladder (bfs->waypoint->astar->iw1->iw2->greedy->rescues).

| Game | Lvl | v13_3 solved | v13_3 time | v13_3 acts | v13_2 solved | v13_2 time | v13_2 acts | winning rung |
|---|---|---|---|---|---|---|---|---|
| ar25 | 0 | YES | 32.6s | 19 | YES | 12.2s | 15 | iw1 |
| ar25 | 1 | YES | 32.4s | 17 | YES | 27.7s | 11 | iw2 |
| ar25 | 2 | no | 603.5s | - | no | 603.3s | - |  |
| ls20 | 0 | YES | 3.0s | 13 | YES | 3.0s | 13 | bfs |
| ls20 | 1 | YES | 76.6s | 45 | YES | 66.1s | 45 | waypoint |
| ls20 | 2 | YES | 77.5s | 39 | YES | 64.1s | 39 | waypoint |
| ls20 | 3 | YES | 150.8s | 43 | YES | 130.3s | 43 | bfs |
| ls20 | 4 | YES | 386.0s | 44 | YES | 367.0s | 44 | bfs |
| ls20 | 5 | no | 600.5s | - | no | 600.7s | - |  |
| ls20 | 6 | no | 600.3s | - | no | 600.3s | - |  |

## Totals

- **v13_3**: 7/10 levels solved, 2563s total search time
- **v13_2**: 7/10 levels solved, 2475s total search time
