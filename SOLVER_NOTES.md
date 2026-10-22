# Booty Dice Solver — Technical Notes

This document covers the design decisions, algorithmic journey, and key findings from building the Booty Dice optimal solver. It's meant to capture the "why" behind the code, including things we tried that didn't work.

---

## Game Overview

Booty Dice is a 2-player dice game. Each player starts with 10 lives, 5 doubloons, and 0 shields. On your turn you roll 6 dice, may freeze any subset and reroll the rest, up to 3 total rolls, then resolve the final dice. **Win by reaching 25 doubloons or killing the opponent.**

**Dice faces:**
- **D (Doubloon):** gain 2D per die
- **X (X Marks the Spot):** 1–2 dice → lose 2X of your doubloons; 3+ dice → Shipwreck (opponent loses X doubloons to buried treasure)
- **J (Jolly Roger):** steal 2J doubloons from opponent
- **C (Cutlass):** deal C damage to opponent (absorbed by their shields first)
- **W (Walk the Plank):** 1–2 dice → lose W lives; 3+ dice → Mutiny (opponent loses W−2 lives)
- **S (Shield):** gain up to S shields from the war chest (shared pool of 30)
- **Blackbeard's Curse (one of each face):** overrides all — opponent loses 2 lives and gives 5 doubloons to buried treasure

**Critical resolution order** (discovered from boardgamearena.com rules; getting this wrong was a material correctness bug in early versions):
1. Blackbeard's Curse check (short-circuits everything)
2. Doubloon (check immediate win at ≥25)
3. X Marks the Spot (only if X < 3)
4. Jolly Roger
5. Cutlass (check immediate win if opponent dies)
6. Walk the Plank (only if W < 3)
7. Shield
8. Mutiny (W ≥ 3)
9. Shipwreck (X ≥ 3, deferred past Jolly Roger)

---

## State Space

A game state from the current player's perspective is a 6-tuple:
```
(my_lives, opp_lives, my_doubloons, opp_doubloons, my_shields, opp_shields)
```

| Dimension | Range | Count |
|-----------|-------|-------|
| lives (each) | 1–10 | 10 |
| doubloons (each) | 0–24 | 25 |
| shield pairs with sum ≤ 30 | — | 496 |

**Total valid states:** 10 × 10 × 25 × 25 × 496 ≈ 31 million

**Storage:** flat float32 array of size 10×10×25×25×31×31 ≈ 60M entries (~229 MB). Invalid shield pairs (sum > 30) are allocated but never read/written.

**Value V(S):** the current player's win probability under optimal play.

**Terminal conditions** (resolved before perspective flip):
- `my_doubloons ≥ 25` or `opp_lives ≤ 0` → V = 1.0 (current player wins)
- `my_lives ≤ 0` → V = 0.0 (current player loses)

---

## Within-Turn DP

Each turn has 3 rolls with 2 rerolls. Given current state S, the value of a turn is:

```
Step 1: For each of 462 possible 6-dice combos d:
          end_val[d] = resolve(d, S)   → 1.0, 0.0, or 1 − V(flip(next_state))

Step 2: keep_ev[K] = Σ_d T[K][d] · end_val[d]   for each of 924 possible keeps K
        (T[K][d] = probability of rolling combo d when keeping K and rerolling the rest)

Step 3: roll2_ev[d] = max_{K ⊆ d} keep_ev[K]    (best keep after second roll)

Step 4: keep_ev[K] = Σ_d T[K][d] · roll2_ev[d]

Step 5: roll1_ev[d] = max_{K ⊆ d} keep_ev[K]    (best keep after first roll)

Step 6: V_new(S) = Σ_d T[∅][d] · roll1_ev[d]   (expected value rolling all 6 fresh)
```

T is a sparse 924×462 matrix (12,376 non-zeros, 2.9% dense). The solver never multiplies by it directly; see *Fast Turn Kernel* below.

### Fast Turn Kernel

The original implementation ran steps 2–5 literally (sparse matvecs plus a scan of every sub-keep of every roll), ~46 µs per state single-threaded. The current kernel computes the same values ~27× faster (~1.7 µs/state):

1. **Reroll one die at a time.** Keeps form a lattice under adding a die. Because dice are independent, `keep_ev[K] = (1/6) Σ_f keep_ev[K + e_f]`, evaluated from size-5 keeps down to the empty keep, with size-6 keeps holding the roll values. That's 462×6 adds instead of 12,376 multiply-adds.
2. **Running max over sub-keeps.** `best[K] = max(keep_ev[K], max_f best[K − e_f])`, evaluated from small keeps up, gives `max_{K ⊆ d} keep_ev[K]` at every roll d in 923×6 maxes instead of 12,376 compare-and-branch steps.
3. **Branch-free resolution.** Each combo's effect is precomputed as per-face parameters (gold, X loss, steal, cutlass, plank, shield, mutiny, shipwreck); Blackbeard's Curse becomes mutiny=2, shipwreck=5. Resolving a state is then min/subtract arithmetic with no early returns.
4. **SIMD across states.** All of the above is identical for every state, so 16 states go through it together as two 8-wide float/int vectors (GCC vector extensions; AVX2 or wider).

Over all 31M states of the converged table, the new kernel matches the original to within 5.5e-6 (float summation order), and both solvers reproduce the original per-layer iteration counts and V values exactly.

---

## Value Iteration — What We Tried

### Approach 1: Global Gauss-Seidel (abandoned)

Iterate over all 31M states in a fixed order, updating V in place. The natural ordering — states closer to a win first — speeds convergence by propagating information downward quickly.

**Problem:** The global fixed-point has spectral radius ρ ≈ 0.98–0.99 because transitions often leave all lives unchanged. Convergence required 50+ iterations × full-state sweep each, and the early (nearly-terminal) states were being re-solved unnecessarily.

### Approach 2: Layer-by-Layer Decomposition

**Key insight:** lives only decrease through transitions (Walk the Plank, Mutiny, Cutlass, Blackbeard's Curse). After the perspective flip, the next state has lives `(opp_new, my_new)` where `opp_new ≤ my_old` and `my_new ≤ opp_old`. So the "layer" containing all states with unordered lives pair `{a, b}` only self-references within the layer (when no lives change) and references lower layers (when lives decrease, already solved exactly).

Process layers in order of `(max(a,b), min(a,b))`:
- **(1,1) → (1,2) → (2,2) → (1,3) → ...** up to **(10,10)**
- 55 layers total, each containing ≤ 620,000 states

**Benefits:**
- Lower layers provide exact boundary conditions
- Each layer's spectral radius = P(no life change in a turn) rather than the global ρ
- For layer (10,10) where lives are plentiful, ρ is low → converges in ~8 iterations
- For layer (1,1) where every W die kills you, ρ is high → needs many iterations

### Approach 3: Anderson(10) Mixing — First Attempt

Added Anderson acceleration to the Jacobi iteration within each layer. The idea: keep the last m Bellman evaluates `G[0..m−1]` and residuals `F[i] = G[i] − V_i`, then solve for mixing weights β to minimise `‖Σ β_j f_j‖²` subject to `Σ β_j = 1` (Lagrange-augmented (m+1)×(m+1) system).

**First attempt result:** Layer (1,1) improved dramatically: 87 iterations down from hitting the 1000-iter cap. But layer (1,5) stalled at 800+ iterations with no convergence sign.

**Root cause:** The (1,b) layers (1 life vs many) have spectral radius ρ ≈ 0.999 because both players aggressively avoid Walk the Plank — player 1 because any W die is instantly fatal, player 2 because optimal play also avoids life loss. With ρ = 0.999 and threshold 1e-6, Jacobi needs ~16,000 iterations; Anderson(10) still needs ~1,000+.

Compounding the problem: after many iterations with nearly-parallel residuals, the 10×10 Gram matrix `H[i][j] = ⟨f_i, f_j⟩` becomes near-singular. With our initial Tikhonov regularisation ε = 1e-10 vs diagonal entries H[i][i] ≈ 62, the relative regularisation was ~1.6×10⁻¹², effectively zero. This caused the mixing weights β to become numerically explosive (‖β‖₁ >> 1), making the mixed iterate far outside [0,1] before clamping, and breaking convergence.

### Approach 4: Anderson(10) Fixed — Final Solver

Three fixes:

1. **Relative Tikhonov regularisation:** `ε = 1e-6 × mean(diag(H))` instead of the absolute 1e-10. This caps the condition number at ~10⁶ regardless of residual magnitude.

2. **β-norm safeguard:** if `‖β‖₁ > 20`, the mixing is numerically garbage — fall back to pure Jacobi (`β = [1, 0, ..., 0]`).

3. **Relaxed convergence threshold:** 1e-4 instead of 1e-6. A 0.01% error in win probability has zero effect on strategy (the optimal keep is unchanged for any state where the win probabilities of competing options differ by more than the rounding error). With ρ = 0.999 and threshold 1e-4, Anderson(10) converges in ~10–12 iterations instead of ~1,000.

**Result:** All 55 layers converge in 7–13 iterations each. Total compute: 172s on 4 cores with the fast turn kernel (originally 1,469s on 20 threads).

---

## Solver Architecture

```
main()
 ├─ build_dice_tables()          enumerate 462 combos, 924 keeps
 ├─ build_transition_matrix()    sparse T[keep → combo] with multinomial probs
 ├─ for each layer {a,b}:
 │   ├─ build lstates[]          flat list of (flat_index, State) for this layer
 │   └─ for iter = 0..MAX_ITERS:
 │       ├─ Jacobi sweep (OMP)   G_slot[i] = turn_values_batch(16 states, V)
 │       │                       F_slot[i] = G_slot[i] − V[idx]
 │       ├─ check convergence    max|F| < 1e-4 → break
 │       ├─ build Gram matrix    H[i][j] = ⟨F[slot_i], F[slot_j]⟩  (serial)
 │       ├─ aug_solve()          (p+1)×(p+1) Lagrange system, Gaussian elim
 │       └─ apply mix (OMP)      V[idx] = clamp(Σ β_j G[slot_j], 0, 1)
 └─ save V to booty_dice_optimal.bin
```

**Memory layout:** Anderson history buffers are slot-major: `G_buf[slot * n + i]`. This gives sequential writes during the Jacobi sweep (one slot at a time) and acceptable scatter-gather during the mixing step.

**OpenMP:** the Jacobi sweep is the bottleneck (462 V lookups + ~11K vector ops per 16-state batch), parallelised over batches. The Anderson Gram-matrix dot products are parallelised too. Each layer iteration for a 620K-state layer takes ~0.3s on 4 cores.

---

## Convergence Data (Normal Game)

| Layer | Iters | Notes |
|-------|-------|-------|
| (1,1) | 7 | Hardest per-state layer (ρ ≈ 0.999), Anderson rescues it |
| (1,b) b=2..10 | 8–12 | High ρ because P1 avoids W; Anderson ≈ 10× speedup |
| (a,b) a≥2 | 11–13 | Moderate ρ, converges smoothly |
| (10,10) | 12 | Largest layer (620K states), low ρ ≈ 0.18 for life changes |
| **Total** | **655** | Across all 55 layers |

Misère converged faster (380 total iters, 4–10 per layer) because optimal misère play seeks life-loss, increasing the probability of transitions that exit the current layer.

---

## Verification (`./verify`)

`./verify booty_dice_optimal.bin` (add `--misere` for the misère table) proves how good the solution and every keep decision are. Logs: `logs/verify_normal.log`, `logs/verify_misere.log`.

**Error bound.** Any final roll containing Walk the Plank changes lives (W<3 hurts the roller, W≥3 is Mutiny, Curse hits the opponent), so it leaves the current layer. Whatever the strategy, each die ends on W with probability ≥ (1/6)³, so a turn stays in its layer with probability ≤ ρ = (215/216)⁶ ≈ 0.9726. Each layer's Bellman operator is therefore a contraction, its fixed point is unique, and a layer with residual r satisfies |V − V*| ≤ E_low + r / (1 − ρ), where E_low bounds the lower layers. The verifier re-solves in double precision to r ≈ 5e-13 and gets:

| | Normal | Misère |
|---|---|---|
| V(start) | 0.637692577855 ± 1.3e-9 | 0.568262763853 ± 1.1e-9 |
| Max \|V − V*\| | 1.32e-9 | 1.07e-9 |

The float32 solver tolerance (1e-4) leaves V(start) off by ~1.5e-5 / ~3e-5.

**Decision certificates.** For each of the 28.6B decisions per variant (every state × every roll seen after the 1st or 2nd throw), the verifier checks one of:
1. *Unique optimum:* the best keep beats every other keep by more than 2·|V − V*|.
2. *Exact tie:* every keep within that margin has the same exact value form, c + Σ wₙ·V*(n), where outcomes reaching the same successor merge. Equal forms mean equal true values, so each tied keep is optimal. Forms are compared by hashing mod 2⁶¹−1 with two seeds; a false match has probability ~1e-37 per comparison.

| | Normal | Misère |
|---|---|---|
| Unique optimum | 24,735,099,455 | 27,610,615,584 |
| Exact tie | 3,908,744,128 | 969,668,174 |
| Unresolved (all candidates within 2·err of optimal) | 156,417 | 63,716,242 |

Most unresolved misère cases have gaps of 1e-13 to 1e-20. These are probably ties between *different* successor states that have equal value, which the identical-form test cannot detect.

**Tighter, per-state bounds (`--tight`).** The layer bound charges every state the worst case accumulated over all 55 layers. `--tight` instead:
1. measures each state's residual in long double, giving r ≈ 6e-15 instead of a 1e-13 floating-point allowance;
2. bounds the error per state by the fixed point of e = r + B(e). Here B is the error-propagation operator: the largest expected successor error over all keep strategies, computed with the same DP. True errors satisfy err ≤ r + B(err) and B is a contraction within a layer, so any e with e ≥ r + B(e) bounds the error. The verifier solves for e with Anderson mixing, inflates it, and accepts it only after checking that inequality on every state (otherwise it falls back to the layer bound);
3. uses each state's largest successor bound as its decision margin.

Normal game (`logs/verify_normal_tight.log`): |V − V*| ≤ 5.2e-13 everywhere, so V(start) = 0.637692577855 ± 4.9e-13. Unresolved decisions drop from 156,417 to **557**, all with gaps of 1e-15 to 1e-12.

*The 58 zero-gap cases* were all decisions after the 2nd throw. In each, the top two keeps had identical forms (a proven tie), and a third keep trailed by 1.5e-11 to 1.75e-9, inside the old margin. The ties come from no-op dice:
- With the opponent at **0 doubloons**, Jolly Roger steals nothing and Shipwreck (X≥3) takes nothing. On rolls like XXXXJS, keeping XXX+J and keeping XXXX are interchangeable (51 cases).
- With the opponent at **≤2 doubloons** and the **war chest empty**, a second J and extra Shields are both no-ops, so JCSSS ≡ JJCSS on JJCSSS (7 cases).

With the tighter bounds, the third keep is proven worse, and all 58 are proven exact ties.

**Audit of the shipped float32 tables.** Choices made from the solver's tables (tolerance 1e-4) are provably suboptimal in 4,706,923 normal decisions (max loss 2.4e-4 win probability) and 364,566,925 misère decisions (max loss 1.8e-4). Rounding the refined values to float32 still leaves 61K / 4.7M provably wrong choices (loss ≤ 4e-8). Proven-correct advice therefore needs double-precision values. `./verify --save-f64` writes them (format version 2 below), and `./cli` reads them and marks each piece of advice that is proven optimal.

The float32 fresh-roll probabilities in `T_sparse` sum to 1 + ~1e-8. The verifier computes them exactly instead; otherwise the residual floors at ~1e-8 on near-certain-win states.

---

## Binary File Format

`booty_dice_optimal.bin` and `booty_dice_misere.bin` share the same layout:

```
Offset  Size  Content
0       4     Magic: 0x424F4F54 ("BOOT")
4       4     Version: 1
8       24    Dimensions: int[6] = {10, 10, 25, 25, 31, 31}
32      240M  float32 V-table, row-major with stride order
              idx = (((((l1*10 + l2)*25 + d1)*25 + d2)*31 + s1)*31 + s2)
```

Total file size: 32 + 10×10×25×25×31×31×4 = 240,250,032 bytes (~229 MB).

Invalid entries (s1+s2 > 30) are stored but should never be read; they are left at the initial value 0.5.

**Version 2** (float64, written by `./verify --save-f64`) adds a 16-byte block after the dimensions and stores doubles:

```
Offset  Size  Content
0       4     Magic: 0x424F4F54 ("BOOT")
4       4     Version: 2
8       24    Dimensions: int[6] = {10, 10, 25, 25, 31, 31}
32      4     Variant: int32, 0 = normal, 1 = misère
36      4     Reserved (0)
40      8     Proven bound on |V − V*|: float64
48      480M  float64 V-table, same index order
```

Total file size: 48 + 60,062,500×8 = 480,500,048 bytes (~458 MB). The CLI and simulators read both versions.

---

## Key Numerical Results

### Normal Game
- **V(start) = 0.6377** — first player wins 63.8% of optimally played games
- **V(near-win) = 1.0000** — (10 lives, 24D, 0S vs 1 life, 0D, 0S): guaranteed win ✓
- **V(near-loss) = 0.0004** — symmetric losing state: ~0% ✓

### Misère Game
- **V(start) = 0.5682** — first player wins the misère game 56.8% of the time
- **V(near-normal-win) = 0.000022** — state that's great in normal is terrible in misère ✓
- **V(near-normal-loss) = 0.999996** — state that's terrible in normal is great in misère ✓

### Fairness Analysis — Shields
With symmetric starting shields s for both players:

| s | P1 win prob |
|---|---|
| 0 | 63.8% |
| 3 | 62.3% |
| 7 | 61.7% |
| 15 | 61.7% |

Floor at 61.7%: shields blunt the first-attack asymmetry but can't offset the doubloon head start. Beyond s=7, the war chest (30 total) is depleted enough that additional shields are inert.

### Fairness Analysis — Doubloon Handicap
Starting state (L1=L2=10, D1=5, D2=x, S1=S2=0):

| x | P1 win prob |
|---|---|
| 5 | 63.8% |
| 9 | 56.4% |
| 11 | 50.3% |
| 12 | 49.5% |

**D2=11** is the closest integer to fair. The ~0.8pp jump between 11 and 12 reflects the even/odd parity effect from doubloons always changing in multiples of 2 (Doubloon +2D, Jolly Roger ±2J·D, X Marks ±2X·D).

---

## Misère Strategy Highlights

In the misère game, optimal play inverts almost everything:

| Die | Normal | Misère |
|-----|--------|--------|
| D (Doubloon) | Keep — collect gold | Avoid — approaching 25D = loss |
| W (Walk the Plank) | Avoid — lose lives | Keep 1–2 — lose lives = win |
| W≥3 (Mutiny) | Seek — kills opponent | Avoid — kills opponent = lose |
| C (Cutlass) | Seek — damages opponent | Avoid — opponent dying = lose |
| S (Shield) | Keep — absorbs damage | Avoid — want to be killed |
| X<3 (X Marks) | Avoid — lose own gold | Keep — shed doubloons = good |
| X≥3 (Shipwreck) | Avoid — sends opp gold away | Avoid — want opp to keep their gold |
| J (Jolly Roger) | Keep — steal from opponent | Avoid — opponent keeping gold is good |

The subtlest reversal: **Jolly Roger is bad in misère** — you're taking doubloons from the opponent when you actually want them to accumulate gold and hit 25 first.

---

## Compute Environment

- Machine: 4-core x86-64 (AVX-512 capable), Linux
- Compiler: g++ -O3 -std=c++17 -march=native -fopenmp
- Normal solver: 172s (~2.9 min), 655 total layer-iters (originally 1,469s on 20 threads)
- Misère solver: 102s (~1.7 min), 380 total layer-iters (originally 948s on 20 threads)
- RAM peak: ~700 MB (229 MB V-table + ~50 MB Anderson buffers + misc)
