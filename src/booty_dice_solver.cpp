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

    std::cout << "Transition matrix: " << nnz << " non-zeros ("
              << std::fixed << std::setprecision(1)
              << 100.0 * nnz / ((long long)nk * nc) << "% dense)" << std::endl;
}

// -------------------------------------------------------------------------
// DICE RESOLUTION
// -------------------------------------------------------------------------

// Each combo's effect reduced to per-face parameters, so resolution becomes
// branch-free min/subtract arithmetic that vectorizes across states.
//
// Resolution order (see SOLVER_NOTES.md): Doubloon (win at >= 25), X Marks
// (X < 3), Jolly Roger, Cutlass (win if opponent dies), Walk the Plank (W < 3),
// Shield, Mutiny (W >= 3), Shipwreck (X >= 3). Blackbeard's Curse overrides
// everything; it is encoded as mutiny = 2 (opponent loses 2 lives) and
// ship = 5 (opponent gives up to 5 doubloons), which yields the same result.
struct ComboEffect { int gold, xlose, steal, cut, plank, shield, mutiny, ship; };
static ComboEffect effects[462];

void build_effects() {
    for (int di = 0; di < (int)all_combos.size(); ++di) {
        const int* c = all_combos[di].c;
        ComboEffect& e = effects[di];
        if (c[F_D]==1 && c[F_X]==1 && c[F_J]==1 && c[F_C]==1 && c[F_W]==1 && c[F_S]==1) {
            e = {0, 0, 0, 0, 0, 0, 2, 5};
            continue;
        }
        e.gold   = 2 * c[F_D];
        e.xlose  = c[F_X] < 3 ? 2 * c[F_X] : 0;
        e.steal  = 2 * c[F_J];
        e.cut    = c[F_C];
        e.plank  = c[F_W] < 3 ? c[F_W] : 0;
        e.shield = c[F_S];
        e.mutiny = c[F_W] >= 3 ? c[F_W] - 2 : 0;
        e.ship   = c[F_X] >= 3 ? c[F_X] : 0;
    }
}

// -------------------------------------------------------------------------
// KEEP LATTICE
// -------------------------------------------------------------------------

// Keeps form a lattice under adding/removing one die. Rerolling dice one at a
// time gives keep_ev[K] = (1/6) Σ_f keep_ev[K + e_f], and the best keep within
// a roll d is a running max over K - e_f. Both replace the 12,376-entry
// sparse transition matrix and subset scans with ~6 ops per keep.
static int add_child[924][NUM_FACES];   // K + e_f  (unused when |K| == 6)
static int sub_parent[924][NUM_FACES];  // K - e_f  (K itself when c_f == 0)
static int combo_keep[462];             // keep index of the full 6-dice combo
static std::vector<int> keeps_by_size[NUM_DICE + 1];
static float roll_prob[462];            // P(fresh roll of 6 dice == combo)

void build_lattice() {
    for (int ki = 0; ki < (int)all_keeps.size(); ++ki) {
        const DiceCombo& k = all_keeps[ki];
        int sz = 0;
        for (int f = 0; f < NUM_FACES; ++f) sz += k.c[f];
        keeps_by_size[sz].push_back(ki);
        for (int f = 0; f < NUM_FACES; ++f) {
            DiceCombo t = k; t.c[f]++;
            add_child[ki][f] = sz < NUM_DICE ? keep_key_map[combo_key(t)] : -1;
            t = k; t.c[f]--;
            sub_parent[ki][f] = k.c[f] > 0 ? keep_key_map[combo_key(t)] : ki;
        }
    }
    for (int di = 0; di < (int)all_combos.size(); ++di)
        combo_keep[di] = keep_key_map[combo_key(all_combos[di])];
    for (const auto& e : T_sparse[empty_keep_idx]) roll_prob[e.di] = e.prob;
}

// -------------------------------------------------------------------------
// WITHIN-TURN DP (SIMD across states)
// -------------------------------------------------------------------------

static constexpr int LANES = 8;              // floats per vector (AVX2)
static constexpr int NV    = 2;              // vectors per batch
static constexpr int BATCH = LANES * NV;     // states evaluated together

typedef float vf __attribute__((vector_size(4 * LANES)));
typedef int   vi __attribute__((vector_size(4 * LANES)));
static inline vf vmax(vf a, vf b) { return a > b ? a : b; }
static inline vi vmin(vi a, vi b) { return a < b ? a : b; }

struct LayerState { long long idx; State s; };

// Evaluates the turn value of up to BATCH states at once; every step of the
// within-turn DP is identical across states, so each lane is one state.
void turn_values_batch(const LayerState* ls, int cnt, const float* V, float* out) {
    alignas(64) static thread_local vf E[924][NV];

    vi ml[NV], ol[NV], md[NV], od[NV], ms[NV], os[NV];
    for (int v = 0; v < NV; ++v)
        for (int l = 0; l < LANES; ++l) {
            int i = v * LANES + l;
            const State& s = ls[i < cnt ? i : 0].s;
            ml[v][l] = s.l1 + 1; ol[v][l] = s.l2 + 1;
            md[v][l] = s.d1;     od[v][l] = s.d2;
            ms[v][l] = s.s1;     os[v][l] = s.s2;
        }

    // Step 1: end value of each final roll (opponent's perspective after flip).
    for (int di = 0; di < (int)all_combos.size(); ++di) {
        const ComboEffect& c = effects[di];
        vf* e = E[combo_keep[di]];
        for (int v = 0; v < NV; ++v) {
            vi m_d = md[v] + c.gold;
            vi win = m_d >= WIN_DOUBLOONS;
            m_d -= vmin(m_d, vi{} + c.xlose);
            vi st = vmin(od[v], vi{} + c.steal);
            m_d += st;
            vi o_d = od[v] - st;
            vi ab  = vmin(os[v], vi{} + c.cut);
            vi o_s = os[v] - ab;
            vi wc  = MAX_SHIELDS - ms[v] - o_s;
            vi o_l = ol[v] - (c.cut - ab);
            win |= o_l <= 0;
            vi m_l = ml[v] - c.plank;
            vi m_s = ms[v] + vmin(wc, vi{} + c.shield);
            o_l -= c.mutiny;
            o_d -= vmin(o_d, vi{} + c.ship);
            win |= (m_d >= WIN_DOUBLOONS) | (o_l <= 0);
            vi loss = (m_l <= 0) & ~win;

            // Next state is (o_l, m_l, o_d, m_d, o_s, m_s); fits in int32.
            vi idx = ((((((o_l - 1) * DIM_L + (m_l - 1)) * DIM_D + o_d) * DIM_D + m_d)
                       * DIM_S + o_s) * DIM_S + m_s);
            idx = (win | loss) != 0 ? vi{} : idx;
            vf g;
            for (int l = 0; l < LANES; ++l) g[l] = V[idx[l]];
            vf r = 1.0f - g;
            r = win  != 0 ? vf{} + 1.0f : r;
            r = loss != 0 ? vf{} : r;
            e[v] = r;
        }
    }

    // Steps 2-5: two rerolls, each = expectation over the rerolled dice, then
    // the best keep given the dice just seen.
    for (int pass = 0; pass < 2; ++pass) {
        for (int sz = NUM_DICE - 1; sz >= 0; --sz)
            for (int ki : keeps_by_size[sz]) {
                const int* ch = add_child[ki];
                for (int v = 0; v < NV; ++v)
                    E[ki][v] = ((E[ch[0]][v] + E[ch[1]][v]) + (E[ch[2]][v] + E[ch[3]][v])
                              + (E[ch[4]][v] + E[ch[5]][v])) * (1.0f / 6.0f);
            }
        for (int sz = 1; sz <= NUM_DICE; ++sz)
            for (int ki : keeps_by_size[sz]) {
                const int* p = sub_parent[ki];
                for (int v = 0; v < NV; ++v)
                    E[ki][v] = vmax(vmax(vmax(E[ki][v], E[p[0]][v]), vmax(E[p[1]][v], E[p[2]][v])),
                                    vmax(vmax(E[p[3]][v], E[p[4]][v]), E[p[5]][v]));
            }
    }

    // Step 6: start of turn EV = expectation over a fresh roll of all 6 dice.
    vf acc[NV] = {};
    for (int di = 0; di < (int)all_combos.size(); ++di) {
        const vf* e = E[combo_keep[di]];
        float p = roll_prob[di];
        for (int v = 0; v < NV; ++v) acc[v] += p * e[v];
    }
    for (int i = 0; i < cnt; ++i) out[i] = acc[i / LANES][i % LANES];
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
    std::cout << "=== Booty Dice 2-Player Solver ===" << std::endl;

    auto t0 = std::chrono::high_resolution_clock::now();

    build_dice_tables();
    build_transition_matrix();
    build_effects();
    build_lattice();

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
            int nbatch = (n + BATCH - 1) / BATCH;
            #pragma omp parallel for schedule(dynamic, 4) reduction(max:max_bellman_delta)
            for (int bi = 0; bi < nbatch; ++bi) {
                int i0 = bi * BATCH;
                int cnt = std::min(BATCH, n - i0);
                float gv[BATCH];
                turn_values_batch(&lstates[i0], cnt, V.data(), gv);
                for (int k = 0; k < cnt; ++k) {
                    float old = V[lstates[i0 + k].idx];
                    G_slot[i0 + k] = gv[k];
                    F_slot[i0 + k] = gv[k] - old;
                    float d = std::fabs(gv[k] - old);
                    if (d > max_bellman_delta) max_bellman_delta = d;
                }
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
                    #pragma omp parallel for schedule(static) reduction(+:dot)
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

    // Sanity checks
    State start{9, 9, 5, 5, 0, 0};
    std::cout << "V(start) = " << std::fixed << std::setprecision(6)
              << V[state_index(start)] << "  (expect ~0.5)" << std::endl;

    State near_win{9, 0, 23, 0, 0, 0};   // my 24 doubloons, opp 1 life
    std::cout << "V(near-win) = " << V[state_index(near_win)]
              << "  (expect ~1.0)" << std::endl;

    State near_loss{0, 9, 0, 23, 0, 0};  // my 1 life, opp 24 doubloons
    std::cout << "V(near-loss) = " << V[state_index(near_loss)]
              << "  (expect ~0.0)" << std::endl;

    // Save
    std::cout << "Saving booty_dice_optimal.bin..." << std::endl;
    std::ofstream out("booty_dice_optimal.bin", std::ios::binary);
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
