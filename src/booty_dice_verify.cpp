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
 * Usage: ./verify booty_dice_optimal.bin [--misere] [--write refined.bin]
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

// Fills E[combo_keep[di]] with the end value of every final roll, reading
// next-state values from V (float or double table).
template <class VT>
void end_values(const LayerState* ls, int cnt, const VT* V, vd (*E)[NV]) {
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
            idx = (win | loss) != 0 ? vl{} : idx;
            vd g;
            for (int l = 0; l < LANES; ++l) g[l] = (double)V[idx[l]];
            vd r = 1.0 - g;
            r = win  != 0 ? vd{} + win_val       : r;
            r = loss != 0 ? vd{} + (1 - win_val) : r;
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
void bellman_batch(const LayerState* ls, int cnt, const double* V, double* out) {
    alignas(64) static thread_local vd E[924][NV];
    end_values(ls, cnt, V, E);
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
};

template <class VT>
void decide_batch(const LayerState* ls, int cnt, const VT* V, Decisions& D) {
    end_values(ls, cnt, V, D.E);
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
    std::string in_path, out_path;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--misere") misere = true;
        else if (a == "--write" && i + 1 < argc) out_path = argv[++i];
        else in_path = a;
    }
    if (in_path.empty()) {
        std::cerr << "Usage: " << argv[0] << " TABLE.bin [--misere] [--write OUT.bin]" << std::endl;
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
        for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
            int slot = iter % M_AA;
            double* G_slot = G_buf.data() + (long long)slot * n;
            double* F_slot = F_buf.data() + (long long)slot * n;
            double res = sweep(G_slot);
            ++iters;
            if (res < REFINE_THRESH) break;
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n; ++i) F_slot[i] = G_slot[i] - V[lstates[i].idx];

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
                V[lstates[i].idx] = std::min(1.0, std::max(0.0, mixed));
            }
        }

        // Certified residual of the stored V (re-measured, whatever the loop did).
        double r = sweep(gv_buf.data()) + FP_SLACK;
        double eps = eps_low + r / (1.0 - RHO);
        eps_layer[a-1][b-1] = eps_layer[b-1][a-1] = eps;
        eps_low = std::max(eps_low, eps);

        double elapsed = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - t0).count();
        std::cout << "Layer (" << std::setw(2) << a << "," << std::setw(2) << b << ")"
                  << "  iters=" << std::setw(3) << iters
                  << (iters == MAX_ITERATIONS ? " [CAP]" : "      ")
                  << std::scientific << std::setprecision(2)
                  << "  residual=" << r << "  |V-V*|<=" << eps
                  << std::fixed << std::setprecision(1)
                  << "  elapsed=" << elapsed << "s" << std::endl;
    }

    std::vector<float> Vr;
    if (!out_path.empty()) {
        Vr.assign(V.begin(), V.end());
        save_table(out_path, Vr);
        std::cout << "Wrote refined float32 table to " << out_path << std::endl;
    }

    // Phase 2: certify every decision and audit the float32 table(s).
    std::cout << "Phase 2: certifying decisions..." << std::endl;
    long long n_dec = 0, n_cert = 0, n_tie = 0, n_exact_tie = 0;
    double min_gap = 1.0;
    Audit audit_in, audit_out;
    for (size_t li = 0; li < layers.size(); ++li) {
        auto [a, b] = layers[li];
        const auto& lstates = layer_states[li];
        int n = (int)lstates.size();
        int nbatch = (n + BATCH - 1) / BATCH;
        // Keep values inherit the error bound of the next states (this layer or lower).
        double margin = 2.0 * eps_layer[a-1][b-1] + FP_SLACK;

        #pragma omp parallel reduction(+:n_dec,n_cert,n_tie,n_exact_tie) reduction(min:min_gap)
        {
            auto D  = std::make_unique<Decisions>();
            auto Df = std::make_unique<Decisions>();
            Audit ain, aout;
            #pragma omp for schedule(dynamic, 4)
            for (int bi = 0; bi < nbatch; ++bi) {
                int i0 = bi * BATCH;
                int cnt = std::min(BATCH, n - i0);
                decide_batch(&lstates[i0], cnt, V.data(), *D);
                for (int r = 0; r < 2; ++r)
                    for (int di = 0; di < (int)all_combos.size(); ++di) {
                        int k = combo_keep[di];
                        for (int i = 0; i < cnt; ++i) {
                            double gap = D->B1[r][k][i / LANES][i % LANES]
                                       - D->B2[r][k][i / LANES][i % LANES];
                            ++n_dec;
                            if (gap > margin) {
                                ++n_cert;
                                min_gap = std::min(min_gap, gap);
                            } else {
                                ++n_tie;
                                if (gap == 0.0) ++n_exact_tie;
                            }
                        }
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
    std::cout << "Decisions checked: " << n_dec << std::endl;
    std::cout << "  certified unique optimum (gap > 2*err): " << n_cert
              << "  (smallest certified gap " << min_gap << ")" << std::endl;
    std::cout << "  near-ties (gap <= 2*err, any listed choice within 2*err of optimal): "
              << n_tie << "  of which exact ties: " << n_exact_tie << std::endl;
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
