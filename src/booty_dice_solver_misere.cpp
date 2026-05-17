/**
 * Booty Dice 2-Player Optimal Solver
 *
 * State: (my_lives, opp_lives, my_doubloons, opp_doubloons, my_shields, opp_shields)
 * from the current player's perspective.
 * V(S) = win probability for the current player under optimal play.
 */

#include <algorithm>
#include <array>
#include <omp.h>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

// -------------------------------------------------------------------------
// CONSTANTS
// -------------------------------------------------------------------------

static constexpr int MAX_LIVES     = 10;
static constexpr int WIN_DOUBLOONS = 25;  // reach this to win
static constexpr int MAX_DOUBLOONS = 25;  // state stores 0..24
static constexpr int MAX_SHIELDS   = 30;  // total in supply
static constexpr int NUM_FACES     = 6;
static constexpr int NUM_DICE      = 6;

// Flat array dimensions: L[10] x L[10] x D[25] x D[25] x S[31] x S[31]
static constexpr int DIM_L = MAX_LIVES;
static constexpr int DIM_D = MAX_DOUBLOONS;
static constexpr int DIM_S = MAX_SHIELDS + 1;  // 31

static constexpr long long TOTAL_STATES =
    (long long)DIM_L * DIM_L * DIM_D * DIM_D * DIM_S * DIM_S;

static constexpr float CONVERGENCE_THRESH = 1e-4f;
static constexpr int   MAX_ITERATIONS     = 1000;

// Dice faces
static constexpr int F_D = 0;  // Doubloon
static constexpr int F_X = 1;  // X Marks the Spot
static constexpr int F_J = 2;  // Jolly Roger
static constexpr int F_C = 3;  // Cutlass
static constexpr int F_W = 4;  // Walk the Plank
static constexpr int F_S = 5;  // Shield

// -------------------------------------------------------------------------
// STATE INDEXING
// -------------------------------------------------------------------------

struct State {
    int l1, l2;  // my_lives-1, opp_lives-1 (0..9)
    int d1, d2;  // my_doubloons, opp_doubloons (0..24)
    int s1, s2;  // my_shields, opp_shields (0..30)
};

inline long long state_index(const State& s) {
    return ((((((long long)s.l1 * DIM_L + s.l2) * DIM_D + s.d1) * DIM_D + s.d2)
             * DIM_S + s.s1) * DIM_S + s.s2);
}

inline State index_to_state(long long idx) {
    State s;
    s.s2 = idx % DIM_S; idx /= DIM_S;
    s.s1 = idx % DIM_S; idx /= DIM_S;
    s.d2 = idx % DIM_D; idx /= DIM_D;
    s.d1 = idx % DIM_D; idx /= DIM_D;
    s.l2 = idx % DIM_L; idx /= DIM_L;
    s.l1 = (int)idx;
    return s;
}

inline bool state_valid(const State& s) {
    return s.s1 >= 0 && s.s2 >= 0 && s.s1 + s.s2 <= MAX_SHIELDS;
}

inline State flip_state(const State& s) {
    return {s.l2, s.l1, s.d2, s.d1, s.s2, s.s1};
}

// -------------------------------------------------------------------------
// DICE COMBO / KEEP ENUMERATION
// -------------------------------------------------------------------------

struct DiceCombo { int c[NUM_FACES]; };  // counts per face, sum = 0..NUM_DICE

static std::vector<DiceCombo> all_combos;  // 462: full 6-dice rolls
static std::vector<DiceCombo> all_keeps;   // 924: sub-multisets (0..6 dice)

// Encode a combo as a base-7 integer (max count per face = 6)
inline int combo_key(const DiceCombo& c) {
    return ((((c.c[0]*7 + c.c[1])*7 + c.c[2])*7 + c.c[3])*7 + c.c[4])*7 + c.c[5];
}

static int combo_key_map[7*7*7*7*7*7];
static int keep_key_map[7*7*7*7*7*7];

void enumerate_multisets(int face, int budget, DiceCombo& cur,
                         bool exact, std::vector<DiceCombo>& out) {
    if (face == NUM_FACES) {
        if (!exact || budget == 0) out.push_back(cur);
        return;
    }
    int limit = budget;
    for (int k = 0; k <= limit; ++k) {
        cur.c[face] = k;
        enumerate_multisets(face + 1, budget - k, cur, exact, out);
    }
    cur.c[face] = 0;
}

void build_dice_tables() {
    memset(combo_key_map, -1, sizeof(combo_key_map));
    memset(keep_key_map,  -1, sizeof(keep_key_map));

    DiceCombo tmp{};
    enumerate_multisets(0, NUM_DICE, tmp, true,  all_combos);  // exact 6
    enumerate_multisets(0, NUM_DICE, tmp, false, all_keeps);   // 0..6

    assert((int)all_combos.size() == 462);
    assert((int)all_keeps.size()  == 924);

    for (int i = 0; i < (int)all_combos.size(); ++i)
        combo_key_map[combo_key(all_combos[i])] = i;
    for (int i = 0; i < (int)all_keeps.size(); ++i)
        keep_key_map[combo_key(all_keeps[i])] = i;

    std::cout << "Dice combos: " << all_combos.size()
              << "  Keep sub-multisets: " << all_keeps.size() << std::endl;
}

// -------------------------------------------------------------------------
// SPARSE TRANSITION MATRIX
// -------------------------------------------------------------------------

struct SparseEntry { int di; float prob; };
static std::vector<std::vector<SparseEntry>> T_sparse;  // T_sparse[ki]
static std::vector<std::vector<int>> valid_keeps;        // valid_keeps[di]
static int empty_keep_idx = -1;

void build_transition_matrix() {
    int nc = (int)all_combos.size();
    int nk = (int)all_keeps.size();

    // Build dense first, then sparsify
    std::vector<std::vector<float>> T(nk, std::vector<float>(nc, 0.0f));

    for (int ki = 0; ki < nk; ++ki) {
        const DiceCombo& keep = all_keeps[ki];
        int kept = 0;
        for (int f = 0; f < NUM_FACES; ++f) kept += keep.c[f];
        int reroll = NUM_DICE - kept;

        long long total = 1;
        for (int i = 0; i < reroll; ++i) total *= NUM_FACES;

        for (long long outcome = 0; outcome < total; ++outcome) {
            DiceCombo result = keep;
            long long tmp = outcome;
            for (int r = 0; r < reroll; ++r) {
                result.c[tmp % NUM_FACES]++;
                tmp /= NUM_FACES;
            }
            int ci = combo_key_map[combo_key(result)];
            if (ci >= 0) T[ki][ci] += 1.0f;
        }
        float denom = (float)total;
        for (int ci = 0; ci < nc; ++ci) T[ki][ci] /= denom;
    }

    // Convert to sparse
    T_sparse.resize(nk);
    long long nnz = 0;
    for (int ki = 0; ki < nk; ++ki) {
        for (int ci = 0; ci < nc; ++ci) {
            if (T[ki][ci] > 0.0f) {
                T_sparse[ki].push_back({ci, T[ki][ci]});
                ++nnz;
            }
        }
    }

    // Find empty keep (all zeros = roll all 6 dice)
    DiceCombo empty{};
    empty_keep_idx = keep_key_map[combo_key(empty)];
    assert(empty_keep_idx >= 0);

    // valid_keeps[di] = all ki where all_keeps[ki] ⊆ all_combos[di]
    valid_keeps.resize(nc);
    for (int ci = 0; ci < nc; ++ci) {
        const DiceCombo& combo = all_combos[ci];
        for (int ki = 0; ki < nk; ++ki) {
            const DiceCombo& keep = all_keeps[ki];
            bool ok = true;
            for (int f = 0; f < NUM_FACES; ++f)
                if (keep.c[f] > combo.c[f]) { ok = false; break; }
            if (ok) valid_keeps[ci].push_back(ki);
        }
    }

    std::cout << "Transition matrix: " << nnz << " non-zeros ("
              << std::fixed << std::setprecision(1)
              << 100.0 * nnz / ((long long)nk * nc) << "% dense)" << std::endl;
}

// -------------------------------------------------------------------------
// DICE RESOLUTION
// -------------------------------------------------------------------------

// Sentinels returned by compute_next_idx for terminal outcomes.
static constexpr long long IDX_WIN  = -1;  // current player wins
static constexpr long long IDX_LOSS = -2;  // current player loses

// Returns the flat V-table index of the next state (opponent's perspective),
// or IDX_WIN / IDX_LOSS for terminal outcomes. No V access — pure arithmetic.
inline long long compute_next_idx(const DiceCombo& combo, const State& s) {
    int cD = combo.c[F_D], cX = combo.c[F_X], cJ = combo.c[F_J];
    int cC = combo.c[F_C], cW = combo.c[F_W], cS = combo.c[F_S];

    int ml = s.l1 + 1, ol = s.l2 + 1;
    int md = s.d1,     od = s.d2;
    int ms = s.s1,     os = s.s2;
    int wc = MAX_SHIELDS - ms - os;

    if (cD==1 && cX==1 && cJ==1 && cC==1 && cW==1 && cS==1) {
        // Blackbeard's Curse: overrides all other effects
        ol -= 2;
        int give = od < 5 ? od : 5;
        od -= give;
    } else {
        // Doubloon — immediate win if >= 25
        md += 2 * cD;
        if (md >= WIN_DOUBLOONS) return IDX_WIN;

        // X Marks the Spot (only if not Shipwreck)
        if (cX > 0 && cX < 3) {
            int lose = md < 2*cX ? md : 2*cX;
            md -= lose;
        }

        // Jolly Roger
        {
            int steal = od < 2*cJ ? od : 2*cJ;
            md += steal;
            od -= steal;
        }

        // Cutlass — immediate win if opponent dies
        if (cC > 0) {
            int ab = os < cC ? os : cC;
            os -= ab;
            wc += ab;
            ol -= (cC - ab);
            if (ol <= 0) return IDX_WIN;
        }

        // Walk the Plank (only if not Mutiny)
        if (cW > 0 && cW < 3) ml -= cW;

        // Shield
        {
            int take = wc < cS ? wc : cS;
            ms += take;
        }

        // Mutiny
        if (cW >= 3) ol -= (cW - 2);

        // Shipwreck (deferred past Jolly Roger)
        if (cX >= 3) {
            int give = od < cX ? od : cX;
            od -= give;
        }
    }

    ml = std::max(ml, 0);
    ol = std::max(ol, 0);
    md = std::max(md, 0);
    od = std::max(od, 0);

    if (md >= WIN_DOUBLOONS || ol <= 0) return IDX_WIN;
    if (ml <= 0)                        return IDX_LOSS;

    State next { ol-1, ml-1, od, md, os, ms };
    return state_index(next);
}

// -------------------------------------------------------------------------
// WITHIN-TURN DP
// -------------------------------------------------------------------------

float compute_turn_value(const State& s, const float* V) {
    int nc = (int)all_combos.size();
    int nk = (int)all_keeps.size();

    static thread_local std::vector<float>     end_val, keep_ev, roll2_ev, roll1_ev;
    static thread_local std::vector<long long> nxt;
    end_val.resize(nc);
    keep_ev.resize(nk);
    roll2_ev.resize(nc);
    roll1_ev.resize(nc);
    nxt.resize(nc);

    // Step 1a: compute next_idx for each combo and prefetch V entries.
    // Issuing all prefetches before any V read hides DRAM latency: by the
    // time the gather loop starts, most cache lines are already in flight.
    for (int di = 0; di < nc; ++di) {
        long long n = compute_next_idx(all_combos[di], s);
        nxt[di] = n;
        if (n >= 0)
            __builtin_prefetch(V + n, 0, 1);  // read, L3 hint
    }

    // Step 1b: gather — V entries should have arrived from DRAM by now.
    // Misère: IDX_WIN (reach 25D or kill opponent) is BAD → 0.0;
    //         IDX_LOSS (own lives gone) is GOOD → 1.0.
    for (int di = 0; di < nc; ++di) {
        long long n = nxt[di];
        if      (n == IDX_WIN)  end_val[di] = 0.0f;
        else if (n == IDX_LOSS) end_val[di] = 1.0f;
        else                    end_val[di] = 1.0f - V[n];
    }

    // Step 2: keep_ev2[K] = sparse T[K] · end_val  (expected value if we keep K and reroll)
    for (int ki = 0; ki < nk; ++ki) {
        float acc = 0.0f;
        for (const auto& e : T_sparse[ki])
            acc += e.prob * end_val[e.di];
        keep_ev[ki] = acc;
    }

    // Step 3: roll2_ev[d] = best expected value with 1 reroll left after seeing d.
    // Shortcut: if end_val[d] == 1.0, keeping everything wins immediately.
    for (int di = 0; di < nc; ++di) {
        if (end_val[di] == 1.0f) {
            roll2_ev[di] = 1.0f;
            continue;
        }
        float best = 0.0f;
        for (int ki : valid_keeps[di])
            if (keep_ev[ki] > best) best = keep_ev[ki];
        roll2_ev[di] = best;
    }

    // Step 4: keep_ev1[K] = sparse T[K] · roll2_ev
    for (int ki = 0; ki < nk; ++ki) {
        float acc = 0.0f;
        for (const auto& e : T_sparse[ki])
            acc += e.prob * roll2_ev[e.di];
        keep_ev[ki] = acc;
    }

    // Step 5: roll1_ev[d] = best expected value with 2 rerolls left after seeing d.
    for (int di = 0; di < nc; ++di) {
        if (roll2_ev[di] == 1.0f) {
            roll1_ev[di] = 1.0f;
            continue;
        }
        float best = 0.0f;
        for (int ki : valid_keeps[di])
            if (keep_ev[ki] > best) best = keep_ev[ki];
        roll1_ev[di] = best;
    }

    // Step 6: start of turn EV = T[empty] · roll1_ev (roll all 6 fresh)
    float new_v = 0.0f;
    for (const auto& e : T_sparse[empty_keep_idx])
        new_v += e.prob * roll1_ev[e.di];

    return new_v;
}


// -------------------------------------------------------------------------
// ANDERSON MIXING
// -------------------------------------------------------------------------

static constexpr int M_AA = 10;  // history depth

// Solve (pp×pp) row-major system A·x = b in-place via Gaussian elimination.
static void aug_solve(double* A, double* b, int pp) {
    for (int col = 0; col < pp; ++col) {
        int piv = col;
        for (int row = col+1; row < pp; ++row)
            if (std::fabs(A[row*pp+col]) > std::fabs(A[piv*pp+col])) piv = row;
        if (piv != col) {
            for (int j = 0; j < pp; ++j) std::swap(A[col*pp+j], A[piv*pp+j]);
            std::swap(b[col], b[piv]);
        }
        if (std::fabs(A[col*pp+col]) < 1e-14) continue;
        double inv = 1.0 / A[col*pp+col];
        for (int row = col+1; row < pp; ++row) {
            double fac = A[row*pp+col] * inv;
            for (int j = col; j < pp; ++j) A[row*pp+j] -= fac * A[col*pp+j];
            b[row] -= fac * b[col];
        }
    }
    for (int col = pp-1; col >= 0; --col) {
        if (std::fabs(A[col*pp+col]) < 1e-14) { b[col] = 0.0; continue; }
        b[col] /= A[col*pp+col];
        for (int row = 0; row < col; ++row) b[row] -= A[row*pp+col] * b[col];
    }
}

// -------------------------------------------------------------------------
// VALUE ITERATION
// -------------------------------------------------------------------------

int main() {
    std::cout << "=== Booty Dice 2-Player Misere Solver (both trying to lose) ===" << std::endl;

    auto t0 = std::chrono::high_resolution_clock::now();

    build_dice_tables();
    build_transition_matrix();

    std::cout << "Allocating V table ("
              << TOTAL_STATES * sizeof(float) / (1024*1024) << " MB)..." << std::endl;
    std::vector<float> V(TOTAL_STATES, 0.5f);

    // Layer-by-layer decomposition: lives are monotonically non-increasing through
    // transitions (Walk the Plank, Mutiny, Cutlass, Blackbeard's Curse only reduce
    // lives). After perspective flip, the next state has l1' ≤ l2 and l2' ≤ l1.
    //
    // So layer {(a,b),(b,a)} (all states with lives pair {a,b}) only references:
    //   - its symmetric counterpart (a,b)↔(b,a) when no lives are lost, and
    //   - strictly lower layers (already solved exactly) when lives are lost.
    //
    // Process in order of increasing (max_lives, then min_lives). Each layer's
    // boundary conditions are exact → spectral radius ≈ P(no life lost) ≈ 0.18
    // for the (10,10) tier → converges in ~8 iterations per layer instead of
    // ~50+ for the global fixed-point problem.

    // Anderson mixing history buffers (slot-major: G_buf[slot*n + i]).
    // Max states per layer: 2 * DIM_D * DIM_D * 496 = 620,000.
    static constexpr int MAX_N_LAYER = 2 * DIM_D * DIM_D * 496;
    std::vector<float> G_buf((long long)M_AA * MAX_N_LAYER);  // Bellman evaluations
    std::vector<float> F_buf((long long)M_AA * MAX_N_LAYER);  // residuals g(V) - V

    struct LayerState { long long idx; State s; };
    std::vector<LayerState> lstates;
    lstates.reserve(MAX_N_LAYER);

    // Build layers: {(a,b),(b,a)} for 1 ≤ a ≤ b ≤ MAX_LIVES, ordered by (b,a).
    struct Layer { int a, b; };
    std::vector<Layer> layers;
    layers.reserve(MAX_LIVES * (MAX_LIVES + 1) / 2);
    for (int b = 1; b <= MAX_LIVES; ++b)
        for (int a = 1; a <= b; ++a)
            layers.push_back({a, b});

    std::cout << "Valid states: " << (long long)DIM_L*DIM_L*DIM_D*DIM_D*496
              << "  threads: " << omp_get_max_threads()
              << "  layers: " << layers.size() << std::endl;
    std::cout << "Layer-by-layer value iteration (Anderson(" << M_AA << "))..." << std::endl;

    long long total_iters = 0;
    auto t_vi = std::chrono::high_resolution_clock::now();

    for (const auto& [a, b] : layers) {
        lstates.clear();
        auto push_lp = [&](int l1, int l2) {
            for (int d1 = 0; d1 < DIM_D; ++d1)
            for (int d2 = 0; d2 < DIM_D; ++d2)
            for (int s1 = 0; s1 < DIM_S; ++s1)
            for (int s2 = 0; s2 < DIM_S - s1; ++s2) {
                State st{l1, l2, d1, d2, s1, s2};
                lstates.push_back({state_index(st), st});
            }
        };
        push_lp(a-1, b-1);
        if (a != b) push_lp(b-1, a-1);
        int n = (int)lstates.size();

        int layer_iters = 0;
        for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
            int slot = iter % M_AA;
            float* G_slot = G_buf.data() + (long long)slot * n;
            float* F_slot = F_buf.data() + (long long)slot * n;

            // Jacobi sweep: evaluate g(V) for every layer state in parallel.
            float max_bellman_delta = 0.0f;
            #pragma omp parallel for schedule(dynamic, 64) reduction(max:max_bellman_delta)
            for (int i = 0; i < n; ++i) {
                float gv  = compute_turn_value(lstates[i].s, V.data());
                float old = V[lstates[i].idx];
                G_slot[i] = gv;
                F_slot[i] = gv - old;
                float d = gv - old; if (d < 0) d = -d;
                if (d > max_bellman_delta) max_bellman_delta = d;
            }
            ++layer_iters;
            if (max_bellman_delta < CONVERGENCE_THRESH) break;

            // Anderson(M_AA) mixing: combine last p Bellman evaluations.
            // Solve: min_{β: Σβ=1} ||Σ β_j f_j||² via augmented Lagrangian system.
            int p = (iter + 1 < M_AA) ? (iter + 1) : M_AA;
            int slots[M_AA];
            for (int j = 0; j < p; ++j) slots[j] = (slot - j + M_AA) % M_AA;

            int pp = p + 1;
            double A[(M_AA+1)*(M_AA+1)] = {};
            double rhs[M_AA+1] = {};
            for (int i = 0; i < p; ++i) {
                const float* Fi = F_buf.data() + (long long)slots[i] * n;
                for (int j = i; j < p; ++j) {
                    const float* Fj = F_buf.data() + (long long)slots[j] * n;
                    double dot = 0.0;
                    for (int k = 0; k < n; ++k) dot += (double)Fi[k] * Fj[k];
                    A[i*pp+j] = A[j*pp+i] = dot;
                }
            }
            // Relative Tikhonov: eps = 1e-6 × mean diagonal to prevent ill-conditioning
            double mean_diag = 0.0;
            for (int i = 0; i < p; ++i) mean_diag += A[i*pp+i];
            mean_diag /= p;
            double reg_eps = 1e-6 * (mean_diag > 0.0 ? mean_diag : 1.0);
            for (int i = 0; i < p; ++i) A[i*pp+i] += reg_eps;

            for (int i = 0; i < p; ++i) { A[i*pp+p] = 1.0; A[p*pp+i] = 1.0; }
            rhs[p] = 1.0;
            aug_solve(A, rhs, pp);

            // If β is numerically unstable (||β||₁ >> 1), fall back to plain Jacobi.
            double beta_l1 = 0.0;
            for (int j = 0; j < p; ++j) beta_l1 += std::fabs(rhs[j]);
            if (beta_l1 > 20.0) {
                std::fill(rhs, rhs + pp, 0.0);
                rhs[0] = 1.0;  // use newest Bellman iterate only
            }

            // V_{k+1} = clamp(Σ β_j · G[slots[j]])
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n; ++i) {
                float mixed = 0.0f;
                for (int j = 0; j < p; ++j)
                    mixed += (float)rhs[j] * G_buf[(long long)slots[j] * n + i];
                if (mixed < 0.0f) mixed = 0.0f;
                if (mixed > 1.0f) mixed = 1.0f;
                V[lstates[i].idx] = mixed;
            }
        }
        total_iters += layer_iters;

        bool capped = (layer_iters == MAX_ITERATIONS);
        if (layer_iters > 1 || (a == MAX_LIVES && b == MAX_LIVES)) {
            double elapsed = std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - t_vi).count();
            std::cout << "Layer (" << std::setw(2) << a << "," << std::setw(2) << b << ")"
                      << "  iters=" << std::setw(3) << layer_iters
                      << (capped ? " [CAP]" : "      ")
                      << "  elapsed=" << std::fixed << std::setprecision(1) << elapsed << "s"
                      << std::endl;
        }
    }

    std::cout << "Converged. Total layer-iters=" << total_iters << std::endl;

    // Sanity checks (misère: near-win in normal = near-loss in misère)
    State start{9, 9, 5, 5, 0, 0};
    std::cout << "V(start) = " << std::fixed << std::setprecision(6)
              << V[state_index(start)] << "  (expect ~0.5)" << std::endl;

    State near_normal_win{9, 0, 23, 0, 0, 0};  // I'm about to reach 25D / kill opp → misère LOSS
    std::cout << "V(near-normal-win) = " << V[state_index(near_normal_win)]
              << "  (expect ~0.0 in misere)" << std::endl;

    State near_normal_loss{0, 9, 0, 23, 0, 0};  // I'm about to die → misère WIN
    std::cout << "V(near-normal-loss) = " << V[state_index(near_normal_loss)]
              << "  (expect ~1.0 in misere)" << std::endl;

    // Save
    std::cout << "Saving booty_dice_misere.bin..." << std::endl;
    std::ofstream out("booty_dice_misere.bin", std::ios::binary);
    uint32_t magic = 0x424F4F54;
    uint32_t version = 1;
    int dims[6] = {DIM_L, DIM_L, DIM_D, DIM_D, DIM_S, DIM_S};
    out.write(reinterpret_cast<const char*>(&magic), 4);
    out.write(reinterpret_cast<const char*>(&version), 4);
    out.write(reinterpret_cast<const char*>(dims), sizeof(dims));
    out.write(reinterpret_cast<const char*>(V.data()), TOTAL_STATES * sizeof(float));
    out.close();

    double total = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t0).count();
    std::cout << "Done. Total time: " << std::fixed << std::setprecision(1)
              << total << "s" << std::endl;
    return 0;
}
