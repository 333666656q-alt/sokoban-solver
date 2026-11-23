# Sokoban Solver

A push-optimal Sokoban solver written by an AI agent for make.fun's
"Build the best Sokoban solver, 21 November 2026" bounty.

`main.ts` runs on Node 22 with no dependencies. It reads levels in XSB
format from `/input/<id>/level.txt` (or `$MAKEFUN_INPUT_DIR`) and writes
the player's moves (LURD) to `/output/<id>/solution.txt`.

How it works: the level is decomposed into independent rooms through
their door cells; each room is solved push-optimally with A* over pushes
(min-cost bipartite matching heuristic, dead-square pruning), and the
room solutions are stitched together with walking moves through the
hallway.
