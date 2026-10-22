/**
 * Booty Dice Solution Verifier
 *
 * Refines a solved V-table (float32, from ./solver or ./solver_misere) in
 * double precision, proves an error bound on the refined values, and then
 * certifies every keep decision: at every state and every roll seen after the
 * 1st or 2nd throw, the best keep must beat the runner-up by more than twice
 * the error bound. It also audits the decisions implied by the input table.
 *
 * Error bound: every final roll containing a Walk the Plank die changes lives
 * (W<3 hurts the roller, W>=3 is Mutiny, Curse hits the opponent), so it leaves
 * the current layer. Under any strategy each die ends on W with probability at
 * least (1/6)^3, so a turn stays in the layer with probability at most
 * RHO = (215/216)^6. For a layer with Bellman residual r = max|T(V) - V| and
 * lower layers already accurate to E_low, |V - V*| <= E_low + r / (1 - RHO).
 *
 * Usage: ./verify booty_dice_optimal.bin [--misere] [--save-f64 refined.bin]
 *   --save-f64 writes the refined values as a version-2 (float64) table that
 *   records the variant and the proven error bound; ./cli reads it.
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
#include <memory>
#include <sstream>
#include <string>
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

static constexpr double REFINE_THRESH  = 1e-12;  // target Bellman residual
static constexpr int    MAX_ITERATIONS = 300;
static constexpr double RHO = 0.97262363;         // >= (215/216)^6, see header
static constexpr double FP_SLACK = 1e-13;         // allowance for double rounding

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
static double roll_prob[462];           // P(fresh roll of 6 dice == combo), exact
// face_steps[f] = (K, K - e_f) for keeps with c_f > 0, smallest keeps first.
// Taking the running max one face at a time merges two disjoint sets at each
// step (they differ in c_f), so the top two values stay exact without ids.
static std::vector<std::pair<int, int>> face_steps[NUM_FACES];
static std::vector<int> sub_keeps[462];  // every keep K ⊆ combo d

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
    // Multinomial 6! / prod(c_f!) / 6^6 in double; the float32 T_sparse
    // probabilities sum to 1 + ~1e-8, which would floor the residual there.
    static const int fact[NUM_DICE + 1] = {1, 1, 2, 6, 24, 120, 720};
    for (int di = 0; di < (int)all_combos.size(); ++di) {
        int ways = fact[NUM_DICE];
        for (int f = 0; f < NUM_FACES; ++f) ways /= fact[all_combos[di].c[f]];
        roll_prob[di] = ways / 46656.0;
    }
    for (int f = 0; f < NUM_FACES; ++f)
        for (int sz = 1; sz <= NUM_DICE; ++sz)
            for (int ki : keeps_by_size[sz])
                if (all_keeps[ki].c[f] > 0) face_steps[f].push_back({ki, sub_parent[ki][f]});
    for (int di = 0; di < (int)all_combos.size(); ++di)
        for (int ki = 0; ki < (int)all_keeps.size(); ++ki) {
            bool ok = true;
            for (int f = 0; f < NUM_FACES; ++f)
                if (all_keeps[ki].c[f] > all_combos[di].c[f]) { ok = false; break; }
            if (ok) sub_keeps[di].push_back(ki);
        }
}

// -------------------------------------------------------------------------
// WITHIN-TURN DP IN DOUBLE (SIMD across states)
// -------------------------------------------------------------------------

static constexpr int LANES = 4;              // doubles per vector (AVX2)
static constexpr int NV    = 4;              // vectors per batch
static constexpr int BATCH = LANES * NV;

typedef double    vd __attribute__((vector_size(8 * LANES)));
typedef long long vl __attribute__((vector_size(8 * LANES)));
static inline vd vmax(vd a, vd b) { return a > b ? a : b; }
static inline vd vmin(vd a, vd b) { return a < b ? a : b; }
static inline vl vmin(vl a, vl b) { return a < b ? a : b; }

struct LayerState { long long idx; State s; };

static bool misere = false;

static constexpr long long NEXT_WIN = -1, NEXT_LOSS = -2;

// Fills E[combo_keep[di]] with the end value of every final roll, reading
// next-state values from V (float or double table). If NI is given, it also
// records the successor: its V-table index, or NEXT_WIN / NEXT_LOSS.
// With as_error, V holds error bounds instead: the end value is the
// successor's bound, and 0 for terminal outcomes (which are exact).
template <class VT>
void end_values(const LayerState* ls, int cnt, const VT* V, vd (*E)[NV],
                vl (*NI)[NV] = nullptr, bool as_error = false) {
    vl ml[NV], ol[NV], md[NV], od[NV], ms[NV], os[NV];
    for (int v = 0; v < NV; ++v)
        for (int l = 0; l < LANES; ++l) {
            int i = v * LANES + l;
            const State& s = ls[i < cnt ? i : 0].s;
            ml[v][l] = s.l1 + 1; ol[v][l] = s.l2 + 1;
            md[v][l] = s.d1;     od[v][l] = s.d2;
            ms[v][l] = s.s1;     os[v][l] = s.s2;
        }
    const double win_val = misere ? 0.0 : 1.0;
    for (int di = 0; di < (int)all_combos.size(); ++di) {
        const ComboEffect& c = effects[di];
        vd* e = E[combo_keep[di]];
        for (int v = 0; v < NV; ++v) {
            vl m_d = md[v] + c.gold;
            vl win = m_d >= WIN_DOUBLOONS;
            m_d -= vmin(m_d, vl{} + c.xlose);
            vl st = vmin(od[v], vl{} + c.steal);
            m_d += st;
            vl o_d = od[v] - st;
            vl ab  = vmin(os[v], vl{} + c.cut);
            vl o_s = os[v] - ab;
            vl wc  = MAX_SHIELDS - ms[v] - o_s;
            vl o_l = ol[v] - (c.cut - ab);
            win |= o_l <= 0;
            vl m_l = ml[v] - c.plank;
            vl m_s = ms[v] + vmin(wc, vl{} + c.shield);
            o_l -= c.mutiny;
            o_d -= vmin(o_d, vl{} + c.ship);
            win |= (m_d >= WIN_DOUBLOONS) | (o_l <= 0);
            vl loss = (m_l <= 0) & ~win;
            vl idx = ((((((o_l - 1) * DIM_L + (m_l - 1)) * DIM_D + o_d) * DIM_D + m_d)
                       * DIM_S + o_s) * DIM_S + m_s);
            if (NI)
                NI[combo_keep[di]][v] = win != 0 ? vl{} + NEXT_WIN
                                      : loss != 0 ? vl{} + NEXT_LOSS : idx;
            idx = (win | loss) != 0 ? vl{} : idx;
            vd g;
            for (int l = 0; l < LANES; ++l) g[l] = (double)V[idx[l]];
            vd r = as_error ? g : 1.0 - g;
            r = win  != 0 ? vd{} + (as_error ? 0.0 : win_val)       : r;
            r = loss != 0 ? vd{} + (as_error ? 0.0 : 1 - win_val)   : r;
            e[v] = r;
        }
    }
}

// keep_ev[K] = mean over next die face of keep_ev[K + e_f]
static inline void average_down(vd (*E)[NV]) {
    for (int sz = NUM_DICE - 1; sz >= 0; --sz)
        for (int ki : keeps_by_size[sz]) {
            const int* ch = add_child[ki];
            for (int v = 0; v < NV; ++v)
                E[ki][v] = ((E[ch[0]][v] + E[ch[1]][v]) + (E[ch[2]][v] + E[ch[3]][v])
                          + (E[ch[4]][v] + E[ch[5]][v])) * (1.0 / 6.0);
        }
}

static inline void start_value(vd (*E)[NV], int cnt, double* out) {
    vd acc[NV] = {};
    for (int di = 0; di < (int)all_combos.size(); ++di) {
        const vd* e = E[combo_keep[di]];
        double p = roll_prob[di];
        for (int v = 0; v < NV; ++v) acc[v] += p * e[v];
    }
    for (int i = 0; i < cnt; ++i) out[i] = acc[i / LANES][i % LANES];
}

// Bellman operator T(V) for up to BATCH states (same DP as the solver).
// With as_error it is the error-propagation operator B(e): the largest
// expected successor error over all keep strategies.
void bellman_batch(const LayerState* ls, int cnt, const double* V, double* out,
                   bool as_error = false) {
    alignas(64) static thread_local vd E[924][NV];
    end_values(ls, cnt, V, E, nullptr, as_error);
    for (int pass = 0; pass < 2; ++pass) {
        average_down(E);
        for (int sz = 1; sz <= NUM_DICE; ++sz)
            for (int ki : keeps_by_size[sz]) {
                const int* p = sub_parent[ki];
                for (int v = 0; v < NV; ++v)
                    E[ki][v] = vmax(vmax(vmax(E[ki][v], E[p[0]][v]), vmax(E[p[1]][v], E[p[2]][v])),
                                    vmax(vmax(E[p[3]][v], E[p[4]][v]), E[p[5]][v]));
            }
    }
    start_value(E, cnt, out);
}

// T(V)(s) in long double (64-bit mantissa) for one state, so the residual
// |T(V) - V| is measured with rounding error ~1e-17 instead of ~1e-15.
static long double turn_value_ld(const State& s, const double* V) {
    static thread_local long double E[924];
    static const int fact[NUM_DICE + 1] = {1, 1, 2, 6, 24, 120, 720};
    const long double win_val = misere ? 0.0L : 1.0L;
    for (int di = 0; di < (int)all_combos.size(); ++di) {
        const ComboEffect& c = effects[di];
        int ml = s.l1 + 1, ol = s.l2 + 1, md = s.d1, od = s.d2, ms = s.s1, os = s.s2;
        int m_d = md + c.gold;
        bool win = m_d >= WIN_DOUBLOONS;
        m_d -= std::min(m_d, c.xlose);
        int st = std::min(od, c.steal);
        m_d += st;
        int o_d = od - st;
        int ab = std::min(os, c.cut);
        int o_s = os - ab;
        int wc = MAX_SHIELDS - ms - o_s;
        int o_l = ol - (c.cut - ab);
        win |= o_l <= 0;
        int m_l = ml - c.plank;
        int m_s = ms + std::min(wc, c.shield);
        o_l -= c.mutiny;
        o_d -= std::min(o_d, c.ship);
        win |= (m_d >= WIN_DOUBLOONS) || (o_l <= 0);
        bool loss = (m_l <= 0) && !win;
        long double r;
        if (win) r = win_val;
        else if (loss) r = 1.0L - win_val;
        else r = 1.0L - (long double)V[state_index(State{o_l - 1, m_l - 1, o_d, m_d, o_s, m_s})];
        E[combo_keep[di]] = r;
    }
    for (int pass = 0; pass < 2; ++pass) {
        for (int sz = NUM_DICE - 1; sz >= 0; --sz)
            for (int ki : keeps_by_size[sz]) {
                const int* ch = add_child[ki];
                long double acc = 0;
                for (int f = 0; f < NUM_FACES; ++f) acc += E[ch[f]];
                E[ki] = acc / 6.0L;
            }
        for (int sz = 1; sz <= NUM_DICE; ++sz)
            for (int ki : keeps_by_size[sz])
                for (int f = 0; f < NUM_FACES; ++f) E[ki] = std::max(E[ki], E[sub_parent[ki][f]]);
    }
    long double acc = 0;
    for (int di = 0; di < (int)all_combos.size(); ++di) {
        int ways = fact[NUM_DICE];
        for (int f = 0; f < NUM_FACES; ++f) ways /= fact[all_combos[di].c[f]];
        acc += (long double)ways * E[combo_keep[di]];
    }
    return acc / 46656.0L;
}

// Per-decision detail for one batch. Round 0 = keep after the 2nd throw
// (one reroll left), round 1 = keep after the 1st throw (two rerolls left).
// For roll d (keep index k of the full combo): B1 = best keep value,
// B2 = runner-up over a different keep, ID = argmax keep, KE = every keep's value.
struct Decisions {
    vd E[924][NV];
    vd KE[2][924][NV];
    vd B1[2][924][NV];
    vd B2[2][924][NV];
    vl ID[2][924][NV];
    vl NI[924][NV];  // successor of each final roll (size-6 keeps only)
};

template <class VT>
void decide_batch(const LayerState* ls, int cnt, const VT* V, Decisions& D) {
    end_values(ls, cnt, V, D.E, D.NI);
    for (int r = 0; r < 2; ++r) {
        average_down(D.E);
        auto& KE = D.KE[r]; auto& B1 = D.B1[r]; auto& B2 = D.B2[r]; auto& ID = D.ID[r];
        for (int ki = 0; ki < 924; ++ki)
            for (int v = 0; v < NV; ++v) {
                KE[ki][v] = B1[ki][v] = D.E[ki][v];
                B2[ki][v] = vd{} - 1.0;
                ID[ki][v] = vl{} + ki;
            }
        for (int f = 0; f < NUM_FACES; ++f)
            for (auto [ki, pk] : face_steps[f])
                for (int v = 0; v < NV; ++v) {
                    vd a1 = B1[ki][v], a2 = B2[ki][v], b1 = B1[pk][v], b2 = B2[pk][v];
                    vl m = b1 > a1;
                    B1[ki][v] = m != 0 ? b1 : a1;
                    ID[ki][v] = m != 0 ? ID[pk][v] : ID[ki][v];
                    B2[ki][v] = vmax(vmin(a1, b1), vmax(a2, b2));
                }
        for (int ki : keeps_by_size[NUM_DICE])
            for (int v = 0; v < NV; ++v) D.E[ki][v] = B1[ki][v];
    }
}

// -------------------------------------------------------------------------
// SYMBOLIC TIE PROOFS
// -------------------------------------------------------------------------

// A keep's true value is a linear form: c * 1 + sum_n w_n * V*(n) over
// successor states n, with exact probability weights (outcomes reaching the
// same successor merge). Two keeps with identical forms have equal true values
// whatever V* is, so a numerical tie between them is an exact tie and every
// tied keep is optimal.
//
// Forms are compared by hashing: each atom (the constant, each V*(n), each
// unresolved best-value R(d)) gets a pseudo-random value mod the prime
// P = 2^61 - 1, and a form hashes to its weighted sum. Distinct forms collide
// with probability 1/P per comparison (Schwartz-Zippel); two independent
// seeds make that 1/P^2 ~ 2e-37.
static constexpr uint64_t PRIME = (1ULL << 61) - 1;
static constexpr int NSEED = 2;

static inline uint64_t addm(uint64_t a, uint64_t b) {
    uint64_t s = a + b;
    return s >= PRIME ? s - PRIME : s;
}
static inline uint64_t subm(uint64_t a, uint64_t b) { return a >= b ? a - b : a + PRIME - b; }
static inline uint64_t mulm(uint64_t a, uint64_t b) {
    unsigned __int128 z = (unsigned __int128)a * b;
    uint64_t lo = (uint64_t)z & PRIME, hi = (uint64_t)(z >> 61);
    return addm(lo, hi);
}
static uint64_t powm(uint64_t a, uint64_t e) {
    uint64_t r = 1;
    for (; e; e >>= 1, a = mulm(a, a)) if (e & 1) r = mulm(r, a);
    return r;
}
static inline uint64_t mix64(uint64_t x) {  // splitmix64 finaliser
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}
static inline uint64_t atom_hash(uint64_t domain, uint64_t id, int seed) {
    return mix64(mix64(domain * 0x51ED27F1ULL + seed) ^ id) % PRIME;
}
static constexpr uint64_t DOM_CONST = 1, DOM_STATE = 2, DOM_OPAQUE = 3;

struct Classes {
    long long cert = 0;       // best keep beats every other keep by > 2*err
    long long tie_form = 0;   // all keeps within 2*err have identical forms
    long long unres_zero = 0; // unresolved, computed gap exactly 0
    long long unres_pos = 0;  // unresolved, 0 < gap <= 2*err
    long long gap_hist[20] = {};  // unres_pos by floor(-log10(gap))
    double min_cert_gap = 1.0;
};

// Classifies both decisions rounds of one state (lane v,l of batch D).
static void classify_state(const Decisions& D, int v, int l, long long sidx,
                           double margin, Classes& c) {
    static thread_local uint64_t H[924][NSEED], R[462][NSEED];
    static uint64_t inv6 = powm(6, PRIME - 2);
    const double win_val = misere ? 0.0 : 1.0;
    int nc = (int)all_combos.size();

    for (int di = 0; di < nc; ++di) {
        int k = combo_keep[di];
        long long n = D.NI[k][v][l];
        for (int j = 0; j < NSEED; ++j) {
            uint64_t one = atom_hash(DOM_CONST, 0, j);
            if (n == NEXT_WIN)       H[k][j] = win_val == 1.0 ? one : 0;
            else if (n == NEXT_LOSS) H[k][j] = win_val == 1.0 ? 0 : one;
            else                     H[k][j] = subm(one, atom_hash(DOM_STATE, n, j));
        }
    }
    for (int r = 0; r < 2; ++r) {
        for (int sz = NUM_DICE - 1; sz >= 0; --sz)
            for (int ki : keeps_by_size[sz]) {
                const int* ch = add_child[ki];
                for (int j = 0; j < NSEED; ++j) {
                    uint64_t acc = 0;
                    for (int f = 0; f < NUM_FACES; ++f) acc = addm(acc, H[ch[f]][j]);
                    H[ki][j] = mulm(acc, inv6);
                }
            }
        for (int di = 0; di < nc; ++di) {
            int k = combo_keep[di];
            double b1 = D.B1[r][k][v][l], gap = b1 - D.B2[r][k][v][l];
            if (gap > margin) {
                ++c.cert;
                c.min_cert_gap = std::min(c.min_cert_gap, gap);
                long long best = D.ID[r][k][v][l];
                for (int j = 0; j < NSEED; ++j) R[di][j] = H[best][j];
                continue;
            }
            int first = -1;
            bool same = true;
            for (int ki : sub_keeps[di]) {
                if (b1 - D.KE[r][ki][v][l] > margin) continue;  // provably worse
                if (first < 0) { first = ki; continue; }
                for (int j = 0; j < NSEED; ++j) same &= H[ki][j] == H[first][j];
                if (!same) break;
            }
            if (same) {
                ++c.tie_form;
                for (int j = 0; j < NSEED; ++j) R[di][j] = H[first][j];
            } else {
                if (gap == 0.0) ++c.unres_zero;
                else {
                    ++c.unres_pos;
                    int bin = std::min(19, std::max(0, (int)std::floor(-std::log10(gap))));
                    ++c.gap_hist[bin];
                }
                // The best value here is not pinned to one form: give it its own atom.
                uint64_t id = ((uint64_t)sidx * 2 + r) * 462 + di;
                for (int j = 0; j < NSEED; ++j) R[di][j] = atom_hash(DOM_OPAQUE, id, j);
            }
        }
        for (int di = 0; di < nc; ++di)
            for (int j = 0; j < NSEED; ++j) H[combo_keep[di]][j] = R[di][j];
    }
}

// -------------------------------------------------------------------------
// ANDERSON MIXING
// -------------------------------------------------------------------------

static constexpr int M_AA = 10;

static void aug_solve(double* A, double* b, int pp) {
    for (int col = 0; col < pp; ++col) {
        int piv = col;
        for (int row = col+1; row < pp; ++row)
            if (std::fabs(A[row*pp+col]) > std::fabs(A[piv*pp+col])) piv = row;
        if (piv != col) {
            for (int j = 0; j < pp; ++j) std::swap(A[col*pp+j], A[piv*pp+j]);
            std::swap(b[col], b[piv]);
        }
        if (std::fabs(A[col*pp+col]) < 1e-300) continue;
        double inv = 1.0 / A[col*pp+col];
        for (int row = col+1; row < pp; ++row) {
            double fac = A[row*pp+col] * inv;
            for (int j = col; j < pp; ++j) A[row*pp+j] -= fac * A[col*pp+j];
            b[row] -= fac * b[col];
        }
    }
    for (int col = pp-1; col >= 0; --col) {
        if (std::fabs(A[col*pp+col]) < 1e-300) { b[col] = 0.0; continue; }
        b[col] /= A[col*pp+col];
        for (int row = 0; row < col; ++row) b[row] -= A[row*pp+col] * b[col];
    }
}

// One Anderson(M_AA) step for a layer of n entries. G_buf/F_buf hold the last
// evaluations G = F(X) and residuals G - X (slot-major); X[idx_of(i)] receives
// the mixed iterate, clamped to [lo, hi].
template <class IdxOf>
static void anderson_mix(int iter, int n, const std::vector<double>& G_buf,
                         const std::vector<double>& F_buf, IdxOf idx_of,
                         std::vector<double>& X, double lo, double hi) {
    int slot = iter % M_AA;
    int p = (iter + 1 < M_AA) ? (iter + 1) : M_AA;
    int slots[M_AA];
    for (int j = 0; j < p; ++j) slots[j] = (slot - j + M_AA) % M_AA;
    int pp = p + 1;
    double A[(M_AA+1)*(M_AA+1)] = {};
    double rhs[M_AA+1] = {};
    for (int i = 0; i < p; ++i) {
        const double* Fi = F_buf.data() + (long long)slots[i] * n;
        for (int j = i; j < p; ++j) {
            const double* Fj = F_buf.data() + (long long)slots[j] * n;
            double dot = 0.0;
            #pragma omp parallel for schedule(static) reduction(+:dot)
            for (int k = 0; k < n; ++k) dot += Fi[k] * Fj[k];
            A[i*pp+j] = A[j*pp+i] = dot;
        }
    }
    double mean_diag = 0.0;
    for (int i = 0; i < p; ++i) mean_diag += A[i*pp+i];
    mean_diag /= p;
    double reg_eps = 1e-10 * (mean_diag > 0.0 ? mean_diag : 1.0);
    for (int i = 0; i < p; ++i) A[i*pp+i] += reg_eps;
    for (int i = 0; i < p; ++i) { A[i*pp+p] = 1.0; A[p*pp+i] = 1.0; }
    rhs[p] = 1.0;
    aug_solve(A, rhs, pp);
    double beta_l1 = 0.0;
    for (int j = 0; j < p; ++j) beta_l1 += std::fabs(rhs[j]);
    if (!(beta_l1 <= 20.0)) {
        std::fill(rhs, rhs + pp, 0.0);
        rhs[0] = 1.0;
    }
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        double mixed = 0.0;
        for (int j = 0; j < p; ++j)
            mixed += rhs[j] * G_buf[(long long)slots[j] * n + i];
        X[idx_of(i)] = std::min(hi, std::max(lo, mixed));
    }
}

// -------------------------------------------------------------------------
// I/O
// -------------------------------------------------------------------------

static bool load_table(const std::string& path, std::vector<float>& V) {
    std::ifstream in(path, std::ios::binary);
    uint32_t magic = 0, version = 0;
    int dims[6];
    in.read(reinterpret_cast<char*>(&magic), 4);
    in.read(reinterpret_cast<char*>(&version), 4);
    in.read(reinterpret_cast<char*>(dims), sizeof(dims));
    if (!in || magic != 0x424F4F54 || version != 1) return false;
    V.resize(TOTAL_STATES);
    in.read(reinterpret_cast<char*>(V.data()), TOTAL_STATES * sizeof(float));
    return (bool)in;
}

// Version 2: float64 values, plus variant (0 normal, 1 misère) and the proven
// bound on |V - V*| so readers can tell which decisions are certified.
struct HeaderV2Extra { int32_t variant; int32_t reserved; double err_bound; };

static void save_table_f64(const std::string& path, const std::vector<double>& V,
                           double err_bound) {
    std::ofstream out(path, std::ios::binary);
    uint32_t magic = 0x424F4F54, version = 2;
    int dims[6] = {DIM_L, DIM_L, DIM_D, DIM_D, DIM_S, DIM_S};
    HeaderV2Extra extra{misere ? 1 : 0, 0, err_bound};
    out.write(reinterpret_cast<const char*>(&magic), 4);
    out.write(reinterpret_cast<const char*>(&version), 4);
    out.write(reinterpret_cast<const char*>(dims), sizeof(dims));
    out.write(reinterpret_cast<const char*>(&extra), sizeof(extra));
    out.write(reinterpret_cast<const char*>(V.data()), TOTAL_STATES * sizeof(double));
}

// Reads a version-2 table (or a bare float64 dump from earlier verifier runs).
static bool load_table_f64(const std::string& path, std::vector<double>& V) {
    std::ifstream in(path, std::ios::binary);
    uint32_t magic = 0, version = 0;
    in.read(reinterpret_cast<char*>(&magic), 4);
    in.read(reinterpret_cast<char*>(&version), 4);
    if (magic == 0x424F4F54) {
        int dims[6];
        HeaderV2Extra extra;
        in.read(reinterpret_cast<char*>(dims), sizeof(dims));
        in.read(reinterpret_cast<char*>(&extra), sizeof(extra));
        if (!in || version != 2 || extra.variant != (misere ? 1 : 0)) return false;
    } else {
        in.seekg(0);
    }
    in.read(reinterpret_cast<char*>(V.data()), TOTAL_STATES * sizeof(double));
    return (bool)in;
}

static void save_table(const std::string& path, const std::vector<float>& V) {
    std::ofstream out(path, std::ios::binary);
    uint32_t magic = 0x424F4F54, version = 1;
    int dims[6] = {DIM_L, DIM_L, DIM_D, DIM_D, DIM_S, DIM_S};
    out.write(reinterpret_cast<const char*>(&magic), 4);
    out.write(reinterpret_cast<const char*>(&version), 4);
    out.write(reinterpret_cast<const char*>(dims), sizeof(dims));
    out.write(reinterpret_cast<const char*>(V.data()), TOTAL_STATES * sizeof(float));
}

// -------------------------------------------------------------------------
// MAIN
// -------------------------------------------------------------------------

struct Audit {
    long long differ = 0;    // decisions where this table's keep != refined keep
    long long wrong = 0;     // ... and the refined keep is provably better
    double max_loss = 0.0;   // largest win-probability loss over those decisions
};

int main(int argc, char** argv) {
    std::string in_path, out_path, save_f64, load_f64;
    bool bounds_only = false;
    // --tight: refine further, measure residuals in long double, and bound the
    // error per state (residual accumulated over the rest of the game) instead
    // of per layer.
    bool tight = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--misere") misere = true;
        else if (a == "--write" && i + 1 < argc) out_path = argv[++i];
        else if (a == "--save-f64" && i + 1 < argc) save_f64 = argv[++i];
        else if (a == "--load-f64" && i + 1 < argc) load_f64 = argv[++i];
        else if (a == "--bounds-only") bounds_only = true;
        else if (a == "--tight") tight = true;
        else in_path = a;
    }
    if (in_path.empty()) {
        std::cerr << "Usage: " << argv[0] << " TABLE.bin [--misere] [--write OUT.bin]"
                  << " [--save-f64 V.bin | --load-f64 V.bin] [--bounds-only] [--tight]" << std::endl
                  << "  --load-f64 skips refinement; the error bound is still re-derived"
                  << " from freshly measured residuals." << std::endl;
        return 1;
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    std::cout << "=== Booty Dice Verifier (" << (misere ? "misere" : "normal") << ") ===" << std::endl;
    build_dice_tables();
    build_transition_matrix();
    build_effects();
    build_lattice();

    std::vector<float> Vf;
    if (!load_table(in_path, Vf)) {
        std::cerr << "Cannot read " << in_path << std::endl;
        return 1;
    }
    std::vector<double> V(Vf.begin(), Vf.end());
    if (!load_f64.empty()) {
        if (!load_table_f64(load_f64, V)) {
            std::cerr << "Cannot read " << load_f64 << std::endl;
            return 1;
        }
    }
    const int max_iters = (load_f64.empty() || tight) ? MAX_ITERATIONS : 0;
    const double refine_thresh = tight ? 1e-14 : REFINE_THRESH;
    std::vector<double> R, Eb;  // --tight: per-state residual and proven error bound
    if (tight) { R.assign(TOTAL_STATES, 0.0); Eb.assign(TOTAL_STATES, 0.0); }

    struct Layer { int a, b; };
    std::vector<Layer> layers;
    for (int b = 1; b <= MAX_LIVES; ++b)
        for (int a = 1; a <= b; ++a)
            layers.push_back({a, b});

    static constexpr int MAX_N_LAYER = 2 * DIM_D * DIM_D * 496;
    std::vector<double> G_buf((long long)M_AA * MAX_N_LAYER);
    std::vector<double> F_buf((long long)M_AA * MAX_N_LAYER);
    std::vector<double> gv_buf(MAX_N_LAYER);
    std::vector<std::vector<LayerState>> layer_states(layers.size());
    double eps_layer[MAX_LIVES][MAX_LIVES] = {};  // proven bound on |V - V*|

    // Phase 1: refine each layer to REFINE_THRESH and bound its error.
    std::cout << "Phase 1: double-precision refinement (Anderson(" << M_AA << "))..." << std::endl;
    double eps_low = 0.0;
    for (size_t li = 0; li < layers.size(); ++li) {
        auto [a, b] = layers[li];
        auto& lstates = layer_states[li];
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
        int nbatch = (n + BATCH - 1) / BATCH;

        // Returns max|T(V) - V| over the layer, storing T(V) in out.
        auto sweep = [&](double* out) {
            double res = 0.0;
            #pragma omp parallel for schedule(dynamic, 4) reduction(max:res)
            for (int bi = 0; bi < nbatch; ++bi) {
                int i0 = bi * BATCH;
                int cnt = std::min(BATCH, n - i0);
                bellman_batch(&lstates[i0], cnt, V.data(), out + i0);
                for (int k = 0; k < cnt; ++k)
                    res = std::max(res, std::fabs(out[i0 + k] - V[lstates[i0 + k].idx]));
            }
            return res;
        };

        int iters = 0;
        for (int iter = 0; iter < max_iters; ++iter) {
            int slot = iter % M_AA;
            double* G_slot = G_buf.data() + (long long)slot * n;
            double* F_slot = F_buf.data() + (long long)slot * n;
            double res = sweep(G_slot);
            ++iters;
            if (res < refine_thresh) break;
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n; ++i) F_slot[i] = G_slot[i] - V[lstates[i].idx];
            anderson_mix(iter, n, G_buf, F_buf, [&](int i) { return lstates[i].idx; }, V, 0.0, 1.0);
        }

        // Certified residual of the stored V (re-measured, whatever the loop did).
        double r = sweep(gv_buf.data()) + FP_SLACK;
        double eps = eps_low + r / (1.0 - RHO);
        eps_layer[a-1][b-1] = eps_layer[b-1][a-1] = eps;
        eps_low = std::max(eps_low, eps);

        std::string tight_note;
        if (tight) {
            // Residual in long double; 5e-17 covers its own rounding.
            double r_max = 0.0;
            #pragma omp parallel for schedule(dynamic, 64) reduction(max:r_max)
            for (int i = 0; i < n; ++i) {
                long double t = turn_value_ld(lstates[i].s, V.data());
                double ri = (double)fabsl(t - (long double)V[lstates[i].idx]) * (1 + 1e-12) + 5e-17;
                R[lstates[i].idx] = ri;
                r_max = std::max(r_max, ri);
            }
            // Error bound: the fixed point of e = R + B(e) bounds |V - V*|
            // (true errors satisfy err <= R + B(err), and B is a contraction).
            // Plain iteration from 0 approaches it from below; then inflate and
            // check e >= R + B(e) (a supersolution, hence >= the fixed point).
            auto bsweep = [&](double* out) {
                double chg = 0.0;
                #pragma omp parallel for schedule(dynamic, 4) reduction(max:chg)
                for (int bi = 0; bi < nbatch; ++bi) {
                    int i0 = bi * BATCH;
                    int cnt = std::min(BATCH, n - i0);
                    bellman_batch(&lstates[i0], cnt, Eb.data(), out + i0, true);
                    for (int k = 0; k < cnt; ++k) {
                        out[i0 + k] += R[lstates[i0 + k].idx];
                        chg = std::max(chg, out[i0 + k] - Eb[lstates[i0 + k].idx]);
                    }
                }
                return chg;
            };
            int eit = 0;
            for (; eit < 200; ++eit) {
                double* G_slot = G_buf.data() + (long long)(eit % M_AA) * n;
                double* F_slot = F_buf.data() + (long long)(eit % M_AA) * n;
                bsweep(G_slot);
                double res = 0.0;
                #pragma omp parallel for schedule(static) reduction(max:res)
                for (int i = 0; i < n; ++i) {
                    F_slot[i] = G_slot[i] - Eb[lstates[i].idx];
                    res = std::max(res, std::fabs(F_slot[i]));
                }
                if (res < 1e-4 * r_max) {
                    #pragma omp parallel for schedule(static)
                    for (int i = 0; i < n; ++i) Eb[lstates[i].idx] = G_slot[i];
                    break;
                }
                anderson_mix(eit, n, G_buf, F_buf, [&](int i) { return lstates[i].idx; }, Eb, 0.0, 1.0);
            }
            bool proven = false;
            for (double delta = 1e-2; delta < 2.0 && !proven; delta *= 4) {
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < n; ++i) gv_buf[i] = Eb[lstates[i].idx];  // keep e_k
                for (int i = 0; i < n; ++i) Eb[lstates[i].idx] = gv_buf[i] * (1 + delta) + delta * r_max;
                std::vector<double> G(n);
                double worst = -1.0;  // max of (R + B(e)) * (1 + 1e-12) - e
                bsweep(G.data());
                for (int i = 0; i < n; ++i)
                    worst = std::max(worst, G[i] * (1 + 1e-12) - Eb[lstates[i].idx]);
                proven = worst <= 0.0;
                if (!proven)
                    for (int i = 0; i < n; ++i) Eb[lstates[i].idx] = gv_buf[i];
            }
            double e_max = 0.0;
            if (!proven) {  // fall back to the layer bound, always valid
                for (int i = 0; i < n; ++i) Eb[lstates[i].idx] = eps;
                e_max = eps;
            } else {
                for (int i = 0; i < n; ++i) e_max = std::max(e_max, Eb[lstates[i].idx]);
            }
            std::ostringstream o;
            o << std::scientific << std::setprecision(2) << "  r_ld<=" << r_max
              << "  e_max=" << e_max << (proven ? "" : " [FALLBACK]") << "  e-iters=" << eit;
            tight_note = o.str();
        }

        double elapsed = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - t0).count();
        std::cout << "Layer (" << std::setw(2) << a << "," << std::setw(2) << b << ")"
                  << "  iters=" << std::setw(3) << iters
                  << (iters == MAX_ITERATIONS ? " [CAP]" : "      ")
                  << std::scientific << std::setprecision(2)
                  << "  residual=" << r << "  |V-V*|<=" << eps << tight_note
                  << std::fixed << std::setprecision(1)
                  << "  elapsed=" << elapsed << "s" << std::endl;
    }

    if (!save_f64.empty()) {
        save_table_f64(save_f64, V, eps_low);
        std::cout << "Wrote refined float64 table to " << save_f64 << std::endl;
    }

    std::vector<float> Vr;
    if (!out_path.empty()) {
        Vr.assign(V.begin(), V.end());
        save_table(out_path, Vr);
        std::cout << "Wrote refined float32 table to " << out_path << std::endl;
    }

    if (bounds_only) {
        std::cout << "Max proven error |V - V*|: " << std::scientific << std::setprecision(2)
                  << eps_low << std::endl;
        return 0;
    }

    // Phase 2: certify every decision and audit the float32 table(s).
    std::cout << "Phase 2: certifying decisions..." << std::endl;
    Classes cls;
    Audit audit_in, audit_out;
    for (size_t li = 0; li < layers.size(); ++li) {
        auto [a, b] = layers[li];
        const auto& lstates = layer_states[li];
        int n = (int)lstates.size();
        int nbatch = (n + BATCH - 1) / BATCH;
        // Keep values inherit the error bound of the next states (this layer or lower).
        double margin = 2.0 * eps_layer[a-1][b-1] + FP_SLACK;

        #pragma omp parallel
        {
            auto D  = std::make_unique<Decisions>();
            auto Df = std::make_unique<Decisions>();
            Audit ain, aout;
            Classes c;
            #pragma omp for schedule(dynamic, 4)
            for (int bi = 0; bi < nbatch; ++bi) {
                int i0 = bi * BATCH;
                int cnt = std::min(BATCH, n - i0);
                decide_batch(&lstates[i0], cnt, V.data(), *D);
                for (int i = 0; i < cnt; ++i) {
                    double m = margin;
                    if (tight) {
                        // Every keep value is an average of successor values, so its
                        // error is at most the largest successor bound.
                        double e = 0.0;
                        for (int ki : keeps_by_size[NUM_DICE]) {
                            long long nx = D->NI[ki][i / LANES][i % LANES];
                            if (nx >= 0) e = std::max(e, Eb[nx]);
                        }
                        m = 2.0 * e + 1e-14;  // + rounding of the double keep values
                    }
                    classify_state(*D, i / LANES, i % LANES, lstates[i0 + i].idx, m, c);
                }
                auto audit = [&](const std::vector<float>& T, Audit& au) {
                    decide_batch(&lstates[i0], cnt, T.data(), *Df);
                    for (int r = 0; r < 2; ++r)
                        for (int di = 0; di < (int)all_combos.size(); ++di) {
                            int k = combo_keep[di];
                            for (int i = 0; i < cnt; ++i) {
                                int v = i / LANES, l = i % LANES;
                                long long kf = Df->ID[r][k][v][l];
                                if (kf == D->ID[r][k][v][l]) continue;
                                double loss = D->B1[r][k][v][l] - D->KE[r][kf][v][l];
                                ++au.differ;
                                if (loss > margin) ++au.wrong;
                                au.max_loss = std::max(au.max_loss, loss);
                            }
                        }
                };
                audit(Vf, ain);
                if (!Vr.empty()) audit(Vr, aout);
            }
            #pragma omp critical
            {
                cls.cert += c.cert;
                cls.tie_form += c.tie_form;
                cls.unres_zero += c.unres_zero;
                cls.unres_pos += c.unres_pos;
                for (int h = 0; h < 20; ++h) cls.gap_hist[h] += c.gap_hist[h];
                cls.min_cert_gap = std::min(cls.min_cert_gap, c.min_cert_gap);
                for (auto [dst, src] : {std::pair{&audit_in, &ain}, std::pair{&audit_out, &aout}}) {
                    dst->differ += src->differ;
                    dst->wrong += src->wrong;
                    dst->max_loss = std::max(dst->max_loss, src->max_loss);
                }
            }
        }
    }

    State start{9, 9, 5, 5, 0, 0};
    double eps_start = eps_layer[9][9];
    std::cout << std::setprecision(12) << std::fixed
              << "V(start) = " << V[state_index(start)]
              << std::scientific << std::setprecision(2) << "  +/- " << eps_start << std::endl;
    std::cout << "Max proven error |V - V*|: " << eps_low << std::endl;
    if (tight) {
        double e_all = 0.0;
        for (const auto& ls : layer_states) for (const auto& q : ls) e_all = std::max(e_all, Eb[q.idx]);
        std::cout << "Per-state bounds (--tight): |V - V*|(start) <= " << Eb[state_index(start)]
                  << ", max over states " << e_all << std::endl;
    }
    long long n_dec = cls.cert + cls.tie_form + cls.unres_zero + cls.unres_pos;
    std::cout << "Decisions checked: " << n_dec << std::endl;
    std::cout << "  proven: unique optimum (gap > 2*err):       " << cls.cert
              << "  (smallest gap " << cls.min_cert_gap << ")" << std::endl;
    std::cout << "  proven: exact tie (identical value forms):  " << cls.tie_form << std::endl;
    std::cout << "  unresolved, computed gap exactly 0:         " << cls.unres_zero << std::endl;
    std::cout << "  unresolved, 0 < gap <= 2*err:               " << cls.unres_pos << std::endl;
    for (int h = 0; h < 20; ++h)
        if (cls.gap_hist[h])
            std::cout << "      gap in [1e-" << h + 1 << ", 1e-" << h << "): " << cls.gap_hist[h] << std::endl;
    std::cout << "  (unresolved decisions: every candidate is within 2*err of optimal)" << std::endl;
    auto report = [&](const char* name, const Audit& au) {
        std::cout << name << ": keeps differing from refined = " << au.differ
                  << ", provably suboptimal = " << au.wrong
                  << ", max win-prob loss = " << au.max_loss << std::endl;
    };
    report("Input table   ", audit_in);
    if (!Vr.empty()) report("Refined float32", audit_out);

    double total = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t0).count();
    std::cout << "Done. Total time: " << std::fixed << std::setprecision(1) << total << "s" << std::endl;
    return 0;
}
