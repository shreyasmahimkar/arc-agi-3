# v13 vs v13_1 benchmark

Fresh solvers, no caches/frontiers. 600s search budget per level per version, workers=8, max-states=5000000. v13 = bfs then greedy (budget split); v13_1 = auto ladder (bfs->waypoint->astar->iw1->iw2->greedy->rescues).

| Game | Lvl | v13_3 solved | v13_3 time | v13_3 acts | v13_2 solved | v13_2 time | v13_2 acts | winning rung |
|---|---|---|---|---|---|---|---|---|
| ar25 | 0 | YES | 12.3s | 15 | YES | 12.4s | 15 | iw1 |
| ar25 | 1 | YES | 29.0s | 11 | YES | 29.6s | 11 | iw2 |
| ar25 | 2 | no | 603.6s | - | no | 603.4s | - |  |
| ls20 | 0 | YES | 3.2s | 13 | YES | 4.5s | 13 | bfs |
| ls20 | 1 | YES | 102.5s | 45 | YES | 98.6s | 45 | astar |
| ls20 | 2 | YES | 82.9s | 39 | YES | 107.2s | 39 | bfs |
| ls20 | 3 | YES | 156.8s | 43 | YES | 138.1s | 43 | bfs |
| ls20 | 4 | YES | 413.8s | 44 | YES | 391.1s | 44 | bfs |
| ls20 | 5 | no | 600.8s | - | no | 600.5s | - |  |
| ls20 | 6 | no | 600.4s | - | no | 600.3s | - |  |

## Totals

- **v13_3**: 7/10 levels solved, 2605s total search time
- **v13_2**: 7/10 levels solved, 2586s total search time
