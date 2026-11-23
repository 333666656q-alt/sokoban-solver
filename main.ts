// Sokoban solver for make.fun program/v1 harness.
// Plain JavaScript (Node 22 type-stripping runs it as-is). No wall clock, no
// randomness: all cutoffs are fixed expansion budgets, so the two grading
// runs produce byte-identical output.
//
// Strategy: levels are a hallway plus independent rooms joined by single
// door cells. Decompose at the doors, solve each room push-optimally with
// A* (matching heuristic + dead-square pruning), then stitch the room
// solutions together with BFS walking moves. Fallback: whole-level A*.

import { readFileSync, readdirSync, mkdirSync, writeFileSync, existsSync, statSync } from 'node:fs';
import { join } from 'node:path';

const INF = 1e9;
// dirs: 0=u, 1=d, 2=l, 3=r (opposite of d is d^1)
const DX = [0, 0, -1, 1];
const DY = [-1, 1, 0, 0];
const DCH = ['u', 'd', 'l', 'r'];

// ---------------------------------------------------------------- parsing
function parseLevel(text) {
  const raw = text.replace(/\r/g, '').split('\n');
  while (raw.length && raw[raw.length - 1] === '') raw.pop();
  const H = raw.length;
  let W = 0;
  for (const l of raw) W = Math.max(W, l.length);
  const wall = new Uint8Array(W * H).fill(1);
  const goals = [];
  const goalSet = new Set();
  const boxes = [];
  let player = -1;
  for (let y = 0; y < H; y++) {
    const line = raw[y];
    for (let x = 0; x < line.length; x++) {
      const ch = line[x];
      const id = y * W + x;
      if (ch === '#') continue;
      wall[id] = 0;
      if (ch === '.' || ch === '*' || ch === '+') { goalSet.add(id); goals.push(id); }
      if (ch === '$' || ch === '*') boxes.push(id);
      if (ch === '@' || ch === '+') player = id;
    }
  }
  // neighbor table, -1 = off board
  const NB = new Int32Array(W * H * 4).fill(-1);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const id = y * W + x;
      for (let d = 0; d < 4; d++) {
        const nx = x + DX[d], ny = y + DY[d];
        if (nx >= 0 && nx < W && ny >= 0 && ny < H) NB[id * 4 + d] = ny * W + nx;
      }
    }
  }
  return { W, H, wall, goals, goalSet, boxes, player, NB };
}

// ------------------------------------------------------- player flood fill
// Returns {vis, rep}: vis = reachable stand cells, rep = min cell index.
function flood(P, standAllowed, boxSet, start) {
  const vis = new Uint8Array(P.W * P.H);
  let rep = start;
  if (start < 0 || !standAllowed[start] || boxSet.has(start)) return { vis, rep: -1 };
  const stack = [start];
  vis[start] = 1;
  while (stack.length) {
    const c = stack.pop();
    if (c < rep) rep = c;
    for (let d = 0; d < 4; d++) {
      const n = P.NB[c * 4 + d];
      if (n >= 0 && !vis[n] && standAllowed[n] && !boxSet.has(n)) {
        vis[n] = 1;
        stack.push(n);
      }
    }
  }
  return { vis, rep };
}

// ------------------------------------------------- single-box push distances
// dist[cell] = min pushes to bring one box from cell to the goal, ignoring
// other boxes and player dynamics (stand cell must be standAllowed, box
// cells must be boxAllowed). Admissible building block for the heuristic.
function pushDistances(P, boxAllowed, standAllowed, goal) {
  const dist = new Int32Array(P.W * P.H).fill(INF);
  if (!boxAllowed[goal]) return dist;
  dist[goal] = 0;
  const queue = [goal];
  let head = 0;
  while (head < queue.length) {
    const b = queue[head++];
    for (let d = 0; d < 4; d++) {
      // forward move was: box at p pushed along d to b. p = b - d, stand = p - d.
      const p = P.NB[b * 4 + (d ^ 1)];
      if (p < 0 || !boxAllowed[p] || dist[p] !== INF) continue;
      const stand = P.NB[p * 4 + (d ^ 1)];
      if (stand < 0 || !standAllowed[stand]) continue;
      dist[p] = dist[b] + 1;
      queue.push(p);
    }
  }
  return dist;
}

// ---------------------------------------------------------------- heuristic
function makeHeuristic(P, dists, goals) {
  const n = goals.length;
  return function heuristic(boxes) {
    if (boxes.length !== n) {
      // unequal counts: sum of per-box minima (admissible)
      let s = 0;
      for (const b of boxes) {
        let m = INF;
        for (let j = 0; j < n; j++) m = Math.min(m, dists[j][b]);
        if (m >= INF) return INF;
        s += m;
      }
      return s;
    }
    if (n <= 6) {
      // exact min-cost bipartite matching by permutation
      const cost = boxes.map(b => dists.map(dt => dt[b]));
      let best = INF;
      const perm = Array.from({ length: n }, (_, i) => i);
      const used = new Array(n).fill(false);
      const cur = new Array(n).fill(0);
      (function rec(i) {
        if (best === 0) return;
        if (i === n) {
          let s = 0;
          for (let k = 0; k < n; k++) {
            const c = cost[k][cur[k]];
            if (c >= INF) { s = INF; break; }
            s += c;
          }
          if (s < best) best = s;
          return;
        }
        for (let j = 0; j < n; j++) {
          if (used[j]) continue;
          used[j] = true; cur[i] = j;
          rec(i + 1);
          used[j] = false;
        }
      })(0);
      void perm;
      return best;
    }
    let s = 0;
    for (const b of boxes) {
      let m = INF;
      for (let j = 0; j < n; j++) m = Math.min(m, dists[j][b]);
      if (m >= INF) return INF;
      s += m;
    }
    return s;
  };
}

// ------------------------------------------------------------------ heap
class Heap {
  constructor() { this.a = []; }
  get size() { return this.a.length; }
  push(node) {
    const a = this.a;
    a.push(node);
    let i = a.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (a[p].f <= a[i].f) break;
      [a[p], a[i]] = [a[i], a[p]];
      i = p;
    }
  }
  pop() {
    const a = this.a;
    const top = a[0];
    const last = a.pop();
    if (a.length) {
      a[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1, r = l + 1;
        let m = i;
        if (l < a.length && a[l].f < a[m].f) m = l;
        if (r < a.length && a[r].f < a[m].f) m = r;
        if (m === i) break;
        [a[m], a[i]] = [a[i], a[m]];
        i = m;
      }
    }
    return top;
  }
  peekF() { return this.a.length ? this.a[0].f : INF; }
}

// ------------------------------------------------------------- push A* core
// Solves one (sub)puzzle. Returns {pushes:[{from,dir}], endBoxes, endPlayer,
// expansions, trapping} with push-optimal pushes, or null.
// exitCell >= 0: among optimal solutions prefer one whose final player
// position can walk back to exitCell; trapping=true if none can.
function solvePuzzle(P, opts) {
  const { boxAllowed, standAllowed, boxes0, goals, playerStart, budget, exitCell } = opts;
  const dists = goals.map(g => pushDistances(P, boxAllowed, standAllowed, g));
  const dead = new Uint8Array(P.W * P.H);
  for (let c = 0; c < P.W * P.H; c++) {
    if (!boxAllowed[c] || P.goalSet.has(c)) continue;
    let alive = false;
    for (let j = 0; j < goals.length; j++) if (dists[j][c] < INF) { alive = true; break; }
    if (!alive) dead[c] = 1;
  }
  const heuristic = makeHeuristic(P, dists, goals);
  const startBoxes = [...boxes0].sort((a, b) => a - b);
  for (const b of startBoxes) if (dead[b]) return null;
  const h0 = heuristic(startBoxes);
  if (h0 >= INF) return null;

  const startSet = new Set(startBoxes);
  const f0 = flood(P, standAllowed, startSet, playerStart);
  if (f0.rep < 0) return null;

  const heap = new Heap();
  const best = new Map();
  const startNode = {
    boxes: startBoxes, playerAct: playerStart, g: 0, f: h0,
    parent: null, from: -1, dir: -1,
  };
  const keyOf = (boxes, rep) => boxes.join(',') + '#' + rep;
  best.set(keyOf(startBoxes, f0.rep), 0);
  heap.push(startNode);

  let expansions = 0;
  let bestGoal = null;
  let bestGoalG = INF;
  let extraPops = 0;
  const EXTRA_CAP = 100000;

  const isGoal = (boxes) => boxes.every(b => P.goalSet.has(b));
  const exitable = (node) => {
    if (exitCell < 0) return true;
    const { vis } = flood(P, standAllowed, new Set(node.boxes), node.playerAct);
    return !!vis[exitCell];
  };

  while (heap.size) {
    if (bestGoal && heap.peekF() > bestGoalG) break;
    if (bestGoal && ++extraPops > EXTRA_CAP) break;
    const node = heap.pop();
    expansions++;
    if (expansions > budget) return null;
    // stale check: recompute rep for this node state
    const nodeSet = new Set(node.boxes);
    const fl = flood(P, standAllowed, nodeSet, node.playerAct);
    if (fl.rep < 0) continue;
    const key = keyOf(node.boxes, fl.rep);
    if (best.has(key) && best.get(key) < node.g) continue;

    if (isGoal(node.boxes)) {
      if (node.g < bestGoalG) { bestGoalG = node.g; bestGoal = node; }
      if (node.g === bestGoalG && bestGoal !== node && exitable(node) && !exitable(bestGoal)) {
        bestGoal = node;
      }
      continue;
    }
    if (bestGoal && node.g >= bestGoalG) continue;

    const { vis } = fl;
    for (let i = 0; i < node.boxes.length; i++) {
      const b = node.boxes[i];
      for (let d = 0; d < 4; d++) {
        const t = P.NB[b * 4 + d];
        if (t < 0 || !boxAllowed[t] || nodeSet.has(t) || dead[t]) continue;
        const stand = P.NB[b * 4 + (d ^ 1)];
        if (stand < 0 || !vis[stand]) continue;
        const nb = node.boxes.slice();
        nb[i] = t;
        nb.sort((x, y) => x - y);
        const h = heuristic(nb);
        if (h >= INF) continue;
        const ng = node.g + 1;
        const nfl = flood(P, standAllowed, new Set(nb), b);
        if (nfl.rep < 0) continue;
        const nkey = keyOf(nb, nfl.rep);
        if (best.has(nkey) && best.get(nkey) <= ng) continue;
        best.set(nkey, ng);
        heap.push({
          boxes: nb, playerAct: b, g: ng, f: ng + h,
          parent: node, from: b, dir: d,
        });
      }
    }
  }

  if (!bestGoal) return null;
  const pushes = [];
  for (let n = bestGoal; n && n.parent; n = n.parent) {
    pushes.push({ from: n.from, dir: n.dir });
  }
  pushes.reverse();
  return {
    pushes,
    endBoxes: bestGoal.boxes,
    endPlayer: bestGoal.playerAct,
    expansions,
    trapping: exitCell >= 0 ? !exitable(bestGoal) : false,
  };
}

// ------------------------------------------------------------- BFS walking
// Shortest walk from `from` to `to` avoiding boxes; returns letter string.
function bfsPath(P, boxSet, from, to) {
  if (from === to) return '';
  const prev = new Int32Array(P.W * P.H).fill(-2);
  prev[from] = -1;
  const queue = [from];
  let head = 0;
  while (head < queue.length) {
    const c = queue[head++];
    for (let d = 0; d < 4; d++) {
      const n = P.NB[c * 4 + d];
      if (n < 0 || prev[n] !== -2 || P.wall[n] || boxSet.has(n)) continue;
      prev[n] = c; // store parent cell; move dir recomputed from the delta
      if (n === to) {
        // reconstruct
        const letters = [];
        let cur = to;
        while (cur !== from) {
          const p = prev[cur];
          const delta = cur - p;
          let d;
          if (delta === -P.W) d = 0; else if (delta === P.W) d = 1;
          else if (delta === -1) d = 2; else d = 3;
          letters.push(DCH[d]);
          cur = p;
        }
        letters.reverse();
        return letters.join('');
      }
      queue.push(n);
    }
  }
  return null;
}

// Expand a push sequence into full moves starting from (boxes0, player0).
// Returns {moves, endBoxes, endPlayer} or null if a walk is impossible.
function expandPushes(P, pushes, boxes0, player0) {
  const boxSet = new Set(boxes0);
  let player = player0;
  const out = [];
  for (const push of pushes) {
    const stand = P.NB[push.from * 4 + (push.dir ^ 1)];
    if (stand < 0) return null;
    const walk = bfsPath(P, boxSet, player, stand);
    if (walk === null) return null;
    out.push(walk);
    out.push(DCH[push.dir]);
    boxSet.delete(push.from);
    boxSet.add(push.from + (P.NB[push.from * 4 + push.dir] - push.from));
    player = push.from;
  }
  return { moves: out.join(''), endBoxes: [...boxSet], endPlayer: player };
}

// ------------------------------------------------------------ replay check
function replay(P, moves) {
  const boxSet = new Set(P.boxes);
  let player = P.player;
  let pushes = 0;
  const dirOf = { u: 0, d: 1, l: 2, r: 3 };
  for (const ch of moves) {
    const d = dirOf[ch];
    if (d === undefined) return { legal: false, solved: false, pushes };
    const t = P.NB[player * 4 + d];
    if (t < 0 || P.wall[t]) return { legal: false, solved: false, pushes };
    if (boxSet.has(t)) {
      const tt = P.NB[t * 4 + d];
      if (tt < 0 || P.wall[tt] || boxSet.has(tt)) return { legal: false, solved: false, pushes };
      boxSet.delete(t);
      boxSet.add(tt);
      pushes++;
    }
    player = t;
  }
  const solved = [...boxSet].every(b => P.goalSet.has(b)) && boxSet.size === P.goals.length;
  return { legal: true, solved, pushes };
}

// ------------------------------------------------------------- decomposition
// Finds door cells: floor cells one row above/below the player's hallway
// row, orthogonally adjacent to a hallway-row floor cell, whose removal
// separates a balanced box/goal cluster (counting a box/goal on the door
// cell itself) that touches no hallway-row cell. Smallest clusters first.
function decompose(P) {
  const { W, H, wall, goalSet } = P;
  const py = Math.floor(P.player / W);
  const boxSet = new Set(P.boxes);
  const candidates = [];
  for (let c = 0; c < W * H; c++) {
    if (wall[c]) continue;
    const cy = Math.floor(c / W), cx = c % W;
    if (cy !== py - 1 && cy !== py + 1) continue;
    const approach = py * W + cx;
    if (wall[approach]) continue;
    // components of the floor graph with c removed
    const comp = new Int32Array(W * H).fill(-1);
    let ncomp = 0;
    for (let s = 0; s < W * H; s++) {
      if (wall[s] || s === c || comp[s] !== -1) continue;
      const stack = [s];
      comp[s] = ncomp;
      while (stack.length) {
        const cur = stack.pop();
        for (let d = 0; d < 4; d++) {
          const n = P.NB[cur * 4 + d];
          if (n >= 0 && !wall[n] && n !== c && comp[n] === -1) {
            comp[n] = ncomp;
            stack.push(n);
          }
        }
      }
      ncomp++;
    }
    const playerComp = comp[P.player];
    if (playerComp < 0) continue;
    const away = [];
    let touchesHallRow = false;
    let minX = W, maxX = -1, minY = H, maxY = -1;
    for (let s = 0; s < W * H; s++) {
      if (comp[s] === -1 || comp[s] === playerComp) continue;
      away.push(s);
      const sy = Math.floor(s / W), sx = s % W;
      if (sy === py) touchesHallRow = true;
      if (sx < minX) minX = sx; if (sx > maxX) maxX = sx;
      if (sy < minY) minY = sy; if (sy > maxY) maxY = sy;
    }
    if (!away.length || touchesHallRow) continue;
    const dx = c % W, dy = cy;
    minX = Math.min(minX, dx); maxX = Math.max(maxX, dx);
    minY = Math.min(minY, dy); maxY = Math.max(maxY, dy);
    if (maxX - minX + 1 > 9 || maxY - minY + 1 > 9) continue;
    let nb = 0, ng = 0;
    for (const s of away) {
      if (boxSet.has(s)) nb++;
      if (goalSet.has(s)) ng++;
    }
    if (boxSet.has(c)) nb++;
    if (goalSet.has(c)) ng++;
    if (nb < 1 || nb !== ng) continue;
    candidates.push({ door: c, approach, cells: away });
  }
  candidates.sort((a, b) => a.cells.length - b.cells.length);
  const claimed = new Set();
  const rooms = [];
  for (const cand of candidates) {
    if (claimed.has(cand.door)) continue;
    if (cand.cells.some(s => claimed.has(s))) continue;
    claimed.add(cand.door);
    for (const s of cand.cells) claimed.add(s);
    const cells = new Set(cand.cells);
    cells.add(cand.door);
    const boxes = P.boxes.filter(b => cells.has(b));
    const goals = P.goals.filter(g => cells.has(g));
    rooms.push({ door: cand.door, approach: cand.approach, cells, boxes, goals });
  }
  if (!rooms.length) return null;
  // every box and goal must belong to exactly one room
  let tb = 0, tg = 0;
  for (const r of rooms) { tb += r.boxes.length; tg += r.goals.length; }
  if (tb !== P.boxes.length || tg !== P.goals.length) return null;
  return rooms;
}

// ------------------------------------------------------------ solve instance
function solveInstance(P) {
  if (P.boxes.length === 0) return { moves: '', note: 'no boxes' };
  const stats = { rooms: 0, expansions: 0, mode: 'none' };

  const rooms = decompose(P);
  if (rooms) {
    stats.rooms = rooms.length;
    const solvedRooms = [];
    let ok = true;
    for (const room of rooms) {
      const boxAllowed = new Uint8Array(P.W * P.H);
      const standAllowed = new Uint8Array(P.W * P.H);
      for (const s of room.cells) { boxAllowed[s] = 1; standAllowed[s] = 1; }
      standAllowed[room.approach] = 1;
      const sol = solvePuzzle(P, {
        boxAllowed, standAllowed,
        boxes0: room.boxes, goals: room.goals,
        playerStart: room.approach,
        budget: 1500000,
        exitCell: room.approach,
      });
      stats.expansions += sol ? sol.expansions : 0;
      if (!sol) { ok = false; break; }
      solvedRooms.push({ room, sol });
    }
    if (ok) {
      // non-trapping rooms first, trapping room last
      solvedRooms.sort((a, b) => (a.sol.trapping ? 1 : 0) - (b.sol.trapping ? 1 : 0));
      const boxSet = new Set(P.boxes);
      let player = P.player;
      const out = [];
      let stitchOk = true;
      for (let ri = 0; ri < solvedRooms.length && stitchOk; ri++) {
        const { room, sol } = solvedRooms[ri];
        const walkIn = bfsPath(P, boxSet, player, room.approach);
        if (walkIn === null) { stitchOk = false; break; }
        out.push(walkIn);
        player = room.approach;
        // replay this room's pushes against the live global state
        const roomBoxes = new Set(room.boxes);
        for (const push of sol.pushes) {
          // find the live box position corresponding to this push: the room
          // solver worked in global coordinates, and rooms are independent,
          // so `from` is the live position too.
          const stand = P.NB[push.from * 4 + (push.dir ^ 1)];
          const walk = bfsPath(P, boxSet, player, stand);
          if (walk === null) { stitchOk = false; break; }
          out.push(walk);
          out.push(DCH[push.dir]);
          boxSet.delete(push.from);
          boxSet.add(P.NB[push.from * 4 + push.dir]);
          roomBoxes.delete(push.from);
          roomBoxes.add(P.NB[push.from * 4 + push.dir]);
          player = push.from;
        }
        if (!stitchOk) break;
        const isLast = ri === solvedRooms.length - 1;
        if (!isLast) {
          const walkOut = bfsPath(P, boxSet, player, room.approach);
          if (walkOut === null) { stitchOk = false; break; }
          out.push(walkOut);
          player = room.approach;
        }
      }
      if (stitchOk) {
        const moves = out.join('');
        const chk = replay(P, moves);
        if (chk.legal && chk.solved) {
          stats.mode = 'rooms';
          return { moves, stats, pushes: chk.pushes };
        }
      }
    }
  }

  // fallback: whole-level A*
  const all = new Uint8Array(P.W * P.H);
  for (let c = 0; c < P.W * P.H; c++) if (!P.wall[c]) all[c] = 1;
  const sol = solvePuzzle(P, {
    boxAllowed: all, standAllowed: all,
    boxes0: P.boxes, goals: P.goals,
    playerStart: P.player,
    budget: 600000,
    exitCell: -1,
  });
  if (sol) {
    stats.expansions += sol.expansions;
    const exp = expandPushes(P, sol.pushes, P.boxes, P.player);
    if (exp) {
      const chk = replay(P, exp.moves);
      if (chk.legal && chk.solved) {
        stats.mode = 'whole';
        return { moves: exp.moves, stats, pushes: chk.pushes };
      }
    }
  }
  stats.mode = 'failed';
  return { moves: '', stats, pushes: 0 };
}

// -------------------------------------------------------------------- main
function main() {
  const inputDir = process.env.MAKEFUN_INPUT_DIR || process.argv[2] || '/input';
  const outputDir = process.env.MAKEFUN_OUTPUT_DIR || process.argv[3] || '/output';
  let ids = [];
  try {
    ids = readdirSync(inputDir)
      .filter(name => {
        const p = join(inputDir, name);
        try { return statSync(p).isDirectory() && existsSync(join(p, 'level.txt')); }
        catch { return false; }
      })
      .sort();
  } catch { ids = []; }
  for (const id of ids) {
    let moves = '';
    let stats = null;
    try {
      const text = readFileSync(join(inputDir, id, 'level.txt'), 'utf8');
      const P = parseLevel(text);
      const res = solveInstance(P);
      moves = res.moves;
      stats = res.stats;
    } catch (e) {
      moves = '';
    }
    mkdirSync(join(outputDir, id), { recursive: true });
    writeFileSync(join(outputDir, id, 'solution.txt'), moves);
    if (stats) {
      console.error(`${id}: mode=${stats.mode} rooms=${stats.rooms || 0} expansions=${stats.expansions || 0} moves=${moves.length}`);
    }
  }
}

main();
