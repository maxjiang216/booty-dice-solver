# Booty Dice Optimal Solver

Computes the exact optimal strategy for the 2-player board game **Booty Dice** via dynamic programming over ~31 million game states. Includes an interactive advisor CLI, game simulators, and a misère (both-trying-to-lose) variant.

## Quick Results

| Variant | First-mover win rate | Compute time | Iterations |
|---------|---------------------|--------------|------------|
| Normal  | **63.8%**           | ~3 min (4 cores) | 655 total layer-iters |
| Misère  | **56.8%**           | ~1.7 min (4 cores) | 380 total layer-iters |

The 63.8% first-mover advantage is inherent to the game: going first lets you collect doubloons and attack before the opponent can respond. No equal starting-shield configuration reduces this below ~61.7%; starting the second player with 11 doubloons instead of 5 brings it to ~50.3%.

---

## Prerequisites

- `g++` with C++17 support
- OpenMP (`libgomp`, standard with GCC)

---

## Building

```bash
make all          # build everything
make solver       # normal game solver only
make solver_misere
make cli
make sim
make sim_misere
make verify       # proves error bounds and certifies every keep decision
make clean        # remove compiled binaries (does NOT delete .bin tables)
```

---

## Step 1 — Solve (run once, ~2–3 min each on 4 cores)

```bash
./solver           # writes booty_dice_optimal.bin  (~229 MB)
./solver_misere    # writes booty_dice_misere.bin   (~229 MB)
```

Progress is printed layer by layer (55 layers total). The solver uses all available CPU cores via OpenMP.

### Optional — proven-optimal tables (~13 min each on 4 cores)

The solver stops at a 1e-4 tolerance, which is close but not exact: some of its keep choices are provably suboptimal (by at most ~0.02% win probability). `./verify` re-solves in double precision, proves an error bound, certifies every decision, and can write a float64 table for the CLI:

```bash
./verify booty_dice_optimal.bin --save-f64 booty_dice_optimal_f64.bin            # ~458 MB
./verify booty_dice_misere.bin --misere --save-f64 booty_dice_misere_f64.bin     # ~458 MB
```

Add `--tight` for per-state error bounds (~5e-13 instead of ~1e-9; ~35 min each). See *Verification* in [SOLVER_NOTES.md](SOLVER_NOTES.md) for the method and results.

---

## Step 2a — Interactive CLI Advisor

```bash
./cli booty_dice_optimal_f64.bin             # normal game, proven advice
./cli booty_dice_misere_f64.bin              # misère game, proven advice
./cli booty_dice_optimal.bin                 # normal game, float32 table (approximate)
./cli booty_dice_misere.bin --misere         # misère game, float32 table (approximate)
```

The advisor tells you the optimal dice to freeze at each roll and your win probability. State is always from the **current player's** perspective.

With a float64 table from `./verify`, each piece of advice also says whether it is **proven optimal** (it beats the next-best keep by more than twice the proven error bound) or a near-tie (proven within ~2e-9 of optimal). Float64 tables record their variant. Float32 tables do not, so pass `--misere` for the misère table.

### CLI Commands

| Command | Description |
|---------|-------------|
| `state` | Show current state and win probability |
| `set L1=N L2=N D1=N D2=N S1=N S2=N` | Set state manually (L=lives 1–10, D=doubloons 0–24, S=shields 0–30) |
| `roll DDXJCW` | Enter the 6 dice you rolled; get freeze advice |
| `flip` | Swap my/opponent perspective (use after opponent's turn) |
| `newroll` | Reset to roll phase 1 |
| `reset` | Return to starting state |
| `help` | Show all commands |
| `quit` | Exit |

### Dice Codes

`D` Doubloon · `X` X Marks the Spot · `J` Jolly Roger · `C` Cutlass · `W` Walk the Plank · `S` Shield

### Example Session

```
set L1=10 L2=10 D1=5 D2=5 S1=0 S2=0
roll D D X J C W        ← roll phase 1: get freeze advice
roll D X J J C W        ← roll phase 2 (enter what you actually rolled)
roll D X J J W W        ← roll phase 3: scored, state advances
flip                    ← opponent's turn; flip again when they finish
```

---

## Step 2b — Game Simulator

Simulate optimal-vs-optimal games and write full transcripts to numbered text files.

```bash
./sim  booty_dice_optimal.bin [num_games] [seed]   # normal
./sim_misere booty_dice_misere.bin [num_games] [seed]  # misère
```

**Defaults:** `num_games=3`, `seed=42`

```bash
./sim booty_dice_optimal.bin 10 777     # 10 normal games, seed 777
./sim_misere booty_dice_misere.bin 5    # 5 misère games, seed 42
```

Each game is written to `game_001.txt`, `game_002.txt`, etc. in the current directory.

---

## File Reference

| File | Purpose |
|------|---------|
| `src/booty_dice_solver.cpp` | Normal game solver |
| `src/booty_dice_solver_misere.cpp` | Misère solver |
| `src/booty_dice_cli.cpp` | Interactive CLI advisor |
| `src/booty_dice_sim.cpp` | Normal game simulator |
| `src/booty_dice_sim_misere.cpp` | Misère game simulator |
| `src/booty_dice_verify.cpp` | Error-bound proof, decision certificates, float64 table writer |
| `booty_dice_optimal.bin` | Precomputed normal strategy table (229 MB, not in git) |
| `booty_dice_misere.bin` | Precomputed misère strategy table (229 MB, not in git) |
| `games/normal/` | Sample normal game transcripts |
| `games/misere/` | Sample misère game transcripts |
| `logs/` | Solver convergence and verification logs |

---

## Key Findings

### First-Mover Advantage
At the symmetric starting state (10 lives, 5 doubloons, 0 shields each), the first player wins **63.8%** of optimally played games. This is the inherent advantage of acting first in a dice game where doubloon collection is asymmetric.

### Fairness Handicaps (same shields for both)
Starting shields reduce the advantage but hit a floor:

| Starting shields (each) | P1 win prob |
|---|---|
| 0 (standard rules) | 63.8% |
| 7+ | 61.7% (floor) |

The floor exists because shields only blunt the attack asymmetry; they cannot offset the doubloon-collection head start.

### Doubloon Handicap
Giving the second player extra starting doubloons crosses 50% between D2=11 (50.3%) and D2=12 (49.5%). **Starting P2 with 11 doubloons** is the fairest integer house rule.

### Misère Strategy
In the misère variant, optimal play is almost the mirror image of normal:
- **Keep W dice** (Walk the Plank = lose a life = good), but avoid 3+ W (Mutiny kills the opponent, which is bad)
- **Avoid D dice** (doubloons push you toward 25 = loss)
- **Keep X dice** (X Marks the Spot takes your own doubloons away = good)
- **Avoid C dice** (Cutlass kills the opponent = bad)
- **Avoid S dice** (shields protect you from being killed = you want to die)
