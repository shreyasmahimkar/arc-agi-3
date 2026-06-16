# v13 vs v13_1 benchmark

Fresh solvers, no caches/frontiers. 28s search budget per level per version, workers=3, max-states=300000. v13 = bfs then greedy (budget split); v13_1 = auto ladder (bfs->waypoint->astar->iw1->iw2->greedy->rescues).

| Game | Lvl | v13_1 solved | v13_1 time | v13_1 acts | v13 solved | v13 time | v13 acts | winning rung |
|---|---|---|---|---|---|---|---|---|
| ar25 | 0 | YES | 13.7s | 15 | no | 31.9s | - | iw1 |
| ar25 | 1 | no | 30.0s | - | no | 31.9s | - |  |
| bp35 | 0 | no | 30.7s | - | no | 33.0s | - |  |
| cd82 | 0 | YES | 0.9s | 5 | YES | 0.9s | 5 | bfs |
| cd82 | 1 | YES | 3.3s | 6 | YES | 3.3s | 6 | bfs |
| ls20 | 0 | YES | 1.6s | 13 | YES | 1.6s | 13 | bfs |
| ls20 | 1 | YES | 20.5s | 45 | YES | 10.1s | 45 | bfs |
| ls20 | 2 | YES | 25.2s | 39 | YES | 13.8s | 39 | bfs |
| su15 | 0 | no | 29.6s | - | no | 31.2s | - |  |
| vc33 | 0 | YES | 0.9s | 3 | YES | 0.9s | 3 | bfs |
| vc33 | 1 | YES | 1.3s | 7 | YES | 1.3s | 7 | bfs |

## Totals

- **v13_1**: 8/11 levels solved, 158s total search time
- **v13**: 7/11 levels solved, 160s total search time
