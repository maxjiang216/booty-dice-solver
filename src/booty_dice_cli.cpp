/**
 * Booty Dice Optimal Advisor CLI
 *
 * Loads a strategy table and advises on optimal dice-freezing decisions.
 *
 * Usage: ./cli [TABLE.bin] [--misere]
 *   Version-1 tables (float32, from ./solver) give approximate advice; pass
 *   --misere with booty_dice_misere.bin. Version-2 tables (float64, from
 *   ./verify --save-f64) record their variant and proven error bound, and each
 *   piece of advice says whether the keep is proven optimal.
 *
 * Turn flow: roll (phase 1) → advise K1 → reroll
 *            roll (phase 2) → advise K2 → reroll
 *            roll (phase 3) → score + update state
 *
 * State is always from the *current* player's perspective.
 * After scoring, the state flips to the opponent's perspective automatically.
 * Use 'flip' to flip back after the opponent's turn.
 */

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// ─── Constants (must match solver) ────────────────────────────────────────────

static constexpr int MAX_LIVES     = 10;
static constexpr int WIN_DOUBLOONS = 25;
static constexpr int MAX_DOUBLOONS = 25;
static constexpr int MAX_SHIELDS   = 30;
static constexpr int NUM_FACES     = 6;
static constexpr int NUM_DICE      = 6;
static constexpr int DIM_L = MAX_LIVES;
static constexpr int DIM_D = MAX_DOUBLOONS;
static constexpr int DIM_S = MAX_SHIELDS + 1;
static constexpr long long TOTAL_STATES =
    (long long)DIM_L * DIM_L * DIM_D * DIM_D * DIM_S * DIM_S;
static constexpr long long IDX_WIN  = -1;
static constexpr long long IDX_LOSS = -2;

static constexpr int F_D=0, F_X=1, F_J=2, F_C=3, F_W=4, F_S=5;
static const char  FACE_CHAR[] = "DXJCWS";

// ─── State + Dice types ───────────────────────────────────────────────────────

struct State    { int l1,l2,d1,d2,s1,s2; };
struct DiceCombo { int c[NUM_FACES]; };

inline long long state_index(const State& s) {
    return ((((((long long)s.l1*DIM_L + s.l2)*DIM_D + s.d1)*DIM_D + s.d2)
             *DIM_S + s.s1)*DIM_S + s.s2);
}

inline bool state_valid(const State& s) {
    return s.l1>=0 && s.l1<DIM_L && s.l2>=0 && s.l2<DIM_L &&
           s.d1>=0 && s.d1<DIM_D && s.d2>=0 && s.d2<DIM_D &&
           s.s1>=0 && s.s2>=0 && s.s1+s.s2<=MAX_SHIELDS;
}

inline int combo_key(const DiceCombo& c) {
    return ((((c.c[0]*7+c.c[1])*7+c.c[2])*7+c.c[3])*7+c.c[4])*7+c.c[5];
}

// ─── Global dice tables ───────────────────────────────────────────────────────

static std::vector<DiceCombo> all_combos;  // 462 full 6-dice rolls
static std::vector<DiceCombo> all_keeps;   // 924 sub-multisets (0..6 dice)
static int combo_key_map[7*7*7*7*7*7];
static int keep_key_map [7*7*7*7*7*7];

struct SparseEntry { int di; double prob; };
static std::vector<std::vector<SparseEntry>> T_sparse;   // T_sparse[ki]
static std::vector<std::vector<int>>         valid_keeps; // valid_keeps[di]

void enumerate_multisets(int face, int budget, DiceCombo& cur,
                         bool exact, std::vector<DiceCombo>& out) {
    if (face == NUM_FACES) { if (!exact || budget==0) out.push_back(cur); return; }
    for (int k=0; k<=budget; ++k) { cur.c[face]=k; enumerate_multisets(face+1,budget-k,cur,exact,out); }
    cur.c[face] = 0;
}

void build_dice_tables() {
    memset(combo_key_map, -1, sizeof(combo_key_map));
    memset(keep_key_map,  -1, sizeof(keep_key_map));
    DiceCombo tmp{};
    enumerate_multisets(0, NUM_DICE, tmp, true,  all_combos);
    enumerate_multisets(0, NUM_DICE, tmp, false, all_keeps);
    assert((int)all_combos.size()==462 && (int)all_keeps.size()==924);
    for (int i=0; i<(int)all_combos.size(); ++i) combo_key_map[combo_key(all_combos[i])] = i;
    for (int i=0; i<(int)all_keeps.size();  ++i) keep_key_map [combo_key(all_keeps[i])]  = i;

    int nc=(int)all_combos.size(), nk=(int)all_keeps.size();
    std::vector<std::vector<double>> T(nk, std::vector<double>(nc, 0.0));
    for (int ki=0; ki<nk; ++ki) {
        const DiceCombo& keep = all_keeps[ki];
        int kept=0; for(int f=0;f<NUM_FACES;++f) kept+=keep.c[f];
        int reroll = NUM_DICE - kept;
        long long total=1; for(int i=0;i<reroll;++i) total*=NUM_FACES;
        for (long long out2=0; out2<total; ++out2) {
            DiceCombo res = keep; long long t=out2;
            for (int r=0;r<reroll;++r){res.c[t%NUM_FACES]++;t/=NUM_FACES;}
            int ci = combo_key_map[combo_key(res)];
            if (ci>=0) T[ki][ci]+=1.0;
        }
        double denom=(double)total;  // exact: counts and 6^k are small integers
        for (int ci=0;ci<nc;++ci) T[ki][ci]/=denom;
    }
    T_sparse.resize(nk);
    for (int ki=0;ki<nk;++ki)
        for (int ci=0;ci<nc;++ci)
            if (T[ki][ci]>0.0) T_sparse[ki].push_back({ci, T[ki][ci]});

    valid_keeps.resize(nc);
    for (int ci=0;ci<nc;++ci) {
        const DiceCombo& combo=all_combos[ci];
        for (int ki=0;ki<nk;++ki) {
            const DiceCombo& keep=all_keeps[ki];
            bool ok=true;
            for (int f=0;f<NUM_FACES;++f) if(keep.c[f]>combo.c[f]){ok=false;break;}
            if (ok) valid_keeps[ci].push_back(ki);
        }
    }
}

// ─── Dice resolution (same as solver) ─────────────────────────────────────────

inline long long compute_next_idx(const DiceCombo& combo, const State& s) {
    int cD=combo.c[F_D],cX=combo.c[F_X],cJ=combo.c[F_J];
    int cC=combo.c[F_C],cW=combo.c[F_W],cS=combo.c[F_S];
    int ml=s.l1+1,ol=s.l2+1,md=s.d1,od=s.d2,ms=s.s1,os=s.s2;
    int wc=MAX_SHIELDS-ms-os;
    if (cD==1&&cX==1&&cJ==1&&cC==1&&cW==1&&cS==1) {
        ol-=2; int give=od<5?od:5; od-=give;
    } else {
        md+=2*cD;
        if(md>=WIN_DOUBLOONS) return IDX_WIN;
        if(cX>0&&cX<3){int lose=md<2*cX?md:2*cX;md-=lose;}
        {int steal=od<2*cJ?od:2*cJ;md+=steal;od-=steal;}
        if(cC>0){int ab=os<cC?os:cC;os-=ab;wc+=ab;ol-=(cC-ab);if(ol<=0)return IDX_WIN;}
        if(cW>0&&cW<3) ml-=cW;
        {int take=wc<cS?wc:cS;ms+=take;}
        if(cW>=3) ol-=(cW-2);
        if(cX>=3){int give=od<cX?od:cX;od-=give;}
    }
    ml=std::max(ml,0); ol=std::max(ol,0); md=std::max(md,0); od=std::max(od,0);
    if(md>=WIN_DOUBLOONS||ol<=0) return IDX_WIN;
    if(ml<=0)                    return IDX_LOSS;
    State next{ol-1,ml-1,od,md,os,ms};
    return state_index(next);
}

// ─── Optimal-keep advisor ─────────────────────────────────────────────────────
//
// Given state S, the combo the player just rolled (combo_di), and how many
// rerolls remain (2 or 1), returns the best keep sub-multiset index, the
// expected win probability achievable from this position, and the value of
// the best different keep (for certifying the choice).

static bool misere = false;  // misère: reaching 25D / killing opp loses, dying wins

struct AdviceResult { int keep_ki; double ev; double runner_up; };

static AdviceResult best_of(const std::vector<double>& keep_ev, int combo_di) {
    AdviceResult r{-1, -1.0, -1.0};
    for (int ki : valid_keeps[combo_di]) {
        if (keep_ev[ki] > r.ev) { r.runner_up = r.ev; r.ev = keep_ev[ki]; r.keep_ki = ki; }
        else if (keep_ev[ki] > r.runner_up) r.runner_up = keep_ev[ki];
    }
    return r;
}

AdviceResult advise(const State& s, int combo_di, int rerolls, const double* V) {
    int nc=(int)all_combos.size(), nk=(int)all_keeps.size();
    std::vector<double> end_val(nc), keep_ev(nk), roll2_ev(nc);
    const double win_val = misere ? 0.0 : 1.0;

    for (int di=0;di<nc;++di) {
        long long n=compute_next_idx(all_combos[di],s);
        if      (n==IDX_WIN)  end_val[di]=win_val;
        else if (n==IDX_LOSS) end_val[di]=1.0-win_val;
        else                  end_val[di]=1.0-V[n];
    }

    // keep_ev[K] = T[K] · end_val (EV of scoring after one more reroll keeping K)
    for (int ki=0;ki<nk;++ki) {
        double acc=0.0;
        for (const auto& e:T_sparse[ki]) acc+=e.prob*end_val[e.di];
        keep_ev[ki]=acc;
    }

    if (rerolls==1) return best_of(keep_ev, combo_di);

    // rerolls==2: need one more layer
    for (int di=0;di<nc;++di) {
        double best=0.0;
        for (int ki:valid_keeps[di]) if(keep_ev[ki]>best) best=keep_ev[ki];
        roll2_ev[di]=best;
    }
    std::vector<double> keep_ev1(nk);
    for (int ki=0;ki<nk;++ki) {
        double acc=0.0;
        for (const auto& e:T_sparse[ki]) acc+=e.prob*roll2_ev[e.di];
        keep_ev1[ki]=acc;
    }
    return best_of(keep_ev1, combo_di);
}

// ─── Parsing and formatting ───────────────────────────────────────────────────

bool parse_dice(const std::string& input, DiceCombo& combo) {
    memset(combo.c, 0, sizeof(combo.c));
    int total=0;
    for (char ch : input) {
        ch = (char)toupper((unsigned char)ch);
        if (ch==' '||ch=='\t'||ch==',') continue;
        int face=-1;
        for (int f=0;f<NUM_FACES;++f) if(ch==FACE_CHAR[f]){face=f;break;}
        if (face<0) return false;
        combo.c[face]++; total++;
    }
    return total==NUM_DICE;
}

std::string combo_str(const DiceCombo& c) {
    std::string s;
    for (int f=0;f<NUM_FACES;++f)
        for (int k=0;k<c.c[f];++k){ s+=FACE_CHAR[f]; s+=' '; }
    if (!s.empty()) s.pop_back();
    return s;
}

// Shows "keep [X X J] · drop [D W S]" style
void print_keep_advice(const DiceCombo& roll, const DiceCombo& keep) {
    std::string sk, sd;
    for (int f=0;f<NUM_FACES;++f) {
        for (int k=0;k<keep.c[f];++k)       { sk+=FACE_CHAR[f]; sk+=' '; }
        for (int k=0;k<roll.c[f]-keep.c[f];++k){ sd+=FACE_CHAR[f]; sd+=' '; }
    }
    if (!sk.empty()) sk.pop_back();
    if (!sd.empty()) sd.pop_back();
    int nkeep=0; for(int f=0;f<NUM_FACES;++f) nkeep+=keep.c[f];
    std::cout << "  Freeze [" << (sk.empty()?"(none)":sk) << "]";
    if (!sd.empty()) std::cout << "  ·  Reroll [" << sd << "]";
    else             std::cout << "  (keep all, no reroll)";
    std::cout << "  (" << nkeep << " kept, " << (NUM_DICE-nkeep) << " rerolled)\n";
}

// Decode a flat state index back to State fields
State decode_state(long long idx) {
    State s;
    s.s2 = idx%DIM_S; idx/=DIM_S;
    s.s1 = idx%DIM_S; idx/=DIM_S;
    s.d2 = idx%DIM_D; idx/=DIM_D;
    s.d1 = idx%DIM_D; idx/=DIM_D;
    s.l2 = idx%DIM_L; idx/=DIM_L;
    s.l1 = (int)idx;
    return s;
}

// ─── Help text ────────────────────────────────────────────────────────────────

static const char* HELP = R"(
Booty Dice Optimal Advisor
==========================
State is always from the CURRENT PLAYER'S perspective.
After scoring, the state flips to the opponent's perspective automatically.

Commands:
  state                    Show state and win probability
  set L1=N L2=N D1=N D2=N S1=N S2=N
                           Set state (L=lives 1-10, D=doubloons 0-24, S=shields 0-30)
  roll DDXJCW              Enter 6 dice for the current roll phase:
                             Phase 1 → advice on what to freeze before reroll 1
                             Phase 2 → advice on what to freeze before reroll 2
                             Phase 3 → score the final dice, advance state
  newroll                  Reset to phase 1 (cancel current roll sequence)
  flip                     Swap my/opponent perspective (use after opponent's turn)
  reset                    Reset to starting state (L=10/10, D=5/5, S=0/0)
  help                     Show this text
  quit / exit              Exit

Dice face codes (case-insensitive, spaces/commas optional):
  D=Doubloon  X=X Marks the Spot  J=Jolly Roger
  C=Cutlass   W=Walk the Plank    S=Shield

Example session:
  set L1=10 L2=10 D1=5 D2=5 S1=0 S2=0
  roll D D X J C W        ← phase 1: freeze recommendation
  roll D X J J C W        ← phase 2 (kept D X J + rolled 3 new dice)
  roll D X J J W W        ← phase 3: scored, state advances
  flip                    ← now it's opponent's turn; flip back when they're done
)";

// ─── Main ─────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    const char* bin_path = "booty_dice_optimal.bin";
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--misere") misere = true;
        else bin_path = argv[i];
    }

    std::cout << "=== Booty Dice Optimal Advisor ===" << std::endl;
    std::cout << "Building dice tables... " << std::flush;
    build_dice_tables();
    std::cout << "done." << std::endl;

    std::cout << "Loading " << bin_path << "... " << std::flush;
    std::vector<double> V(TOTAL_STATES);
    double err_bound = -1.0;  // proven |V - V*| for version-2 tables; unknown otherwise
    {
        std::ifstream in(bin_path, std::ios::binary);
        if (!in) {
            std::cerr << "\nERROR: Cannot open '" << bin_path << "'.\n";
            std::cerr << "Run ./solver first to generate the strategy file.\n";
            return 1;
        }
        uint32_t magic=0, version=0;
        int dims[6]={};
        in.read(reinterpret_cast<char*>(&magic),   4);
        in.read(reinterpret_cast<char*>(&version), 4);
        in.read(reinterpret_cast<char*>(dims),     sizeof(dims));
        if (magic != 0x424F4F54u) {
            std::cerr << "\nERROR: Bad magic number — is this really a Booty Dice table?\n";
            return 1;
        }
        if (version == 1) {
            std::vector<float> Vf(TOTAL_STATES);
            in.read(reinterpret_cast<char*>(Vf.data()), TOTAL_STATES * sizeof(float));
            std::copy(Vf.begin(), Vf.end(), V.begin());
        } else if (version == 2) {
            int32_t variant = 0, reserved = 0;
            in.read(reinterpret_cast<char*>(&variant),   4);
            in.read(reinterpret_cast<char*>(&reserved),  4);
            in.read(reinterpret_cast<char*>(&err_bound), 8);
            if (misere != (variant == 1))
                std::cout << "(table is " << (variant == 1 ? "misère" : "normal")
                          << "; using that) " << std::flush;
            misere = variant == 1;
            in.read(reinterpret_cast<char*>(V.data()), TOTAL_STATES * sizeof(double));
        } else {
            std::cerr << "\nERROR: Unsupported table version " << version << ".\n";
            return 1;
        }
        if (!in) {
            std::cerr << "\nERROR: Truncated file — re-run the solver.\n";
            return 1;
        }
    }
    std::cout << "done." << std::endl;
    std::cout << "Variant: " << (misere ? "misère (both trying to lose)" : "normal") << "\n";
    if (err_bound >= 0.0)
        std::cout << "Values proven accurate to ±" << std::scientific << std::setprecision(2)
                  << err_bound << std::defaultfloat << "; advice is marked when proven optimal.\n";
    else
        std::cout << "float32 table: advice is approximate. For proven advice, use a table\n"
                     "from ./verify --save-f64 (see SOLVER_NOTES.md, Verification).\n";
    std::cout << "Type 'help' for commands.\n" << std::endl;

    State game  = {9, 9, 5, 5, 0, 0};  // L=10/10, D=5/5, S=0/0
    int   phase = 1;                    // current roll phase (1, 2, or 3)

    while (true) {
        // Prompt: show compact state + phase
        std::cout << "[L" << (game.l1+1) << "/" << (game.l2+1)
                  << " D" << game.d1     << "/" << game.d2
                  << " S" << game.s1     << "/" << game.s2
                  << " roll" << phase    << "/3] > " << std::flush;

        std::string line;
        if (!std::getline(std::cin, line)) break;
        while (!line.empty() && isspace((unsigned char)line.front())) line.erase(line.begin());
        while (!line.empty() && isspace((unsigned char)line.back()))  line.pop_back();
        if (line.empty()) continue;

        std::istringstream iss(line);
        std::string cmd; iss >> cmd;
        for (auto& ch : cmd) ch = (char)tolower((unsigned char)ch);

        // ── quit ──
        if (cmd=="quit"||cmd=="exit") break;

        // ── help ──
        if (cmd=="help") { std::cout << HELP; continue; }

        // ── reset ──
        if (cmd=="reset") {
            game={9,9,5,5,0,0}; phase=1;
            std::cout << "Reset to start state.\n";
            continue;
        }

        // ── newroll ──
        if (cmd=="newroll") {
            phase=1;
            std::cout << "Roll sequence reset to phase 1.\n";
            continue;
        }

        // ── flip ──
        if (cmd=="flip") {
            game={game.l2,game.l1,game.d2,game.d1,game.s2,game.s1};
            phase=1;
            std::cout << "Perspective flipped.\n";
            continue;
        }

        // ── state ──
        if (cmd=="state") {
            std::cout << std::fixed << std::setprecision(1);
            std::cout << "  My lives: "     << (game.l1+1)
                      << "   Opp lives: "   << (game.l2+1) << "\n"
                      << "  My doubloons: " << game.d1
                      << "  Opp doubloons: "<< game.d2 << "\n"
                      << "  My shields: "   << game.s1
                      << "   Opp shields: " << game.s2 << "\n";
            if (state_valid(game)) {
                double v = V[state_index(game)];
                std::cout << "  Win probability (start of turn): "
                          << (v*100.0) << "%\n";
            }
            std::cout << "  Roll phase: " << phase << " / 3\n";
            continue;
        }

        // ── set L1=N ... ──
        if (cmd=="set") {
            State ns=game;
            std::string tok;
            bool ok=true;
            while (iss>>tok) {
                auto eq=tok.find('=');
                if (eq==std::string::npos){std::cout<<"Expected KEY=VALUE\n";ok=false;break;}
                std::string key=tok.substr(0,eq);
                for(auto& c:key) c=(char)toupper((unsigned char)c);
                int val;
                try { val=std::stoi(tok.substr(eq+1)); }
                catch(...){std::cout<<"Bad value: "<<tok<<"\n";ok=false;break;}
                if      (key=="L1") ns.l1=val-1;
                else if (key=="L2") ns.l2=val-1;
                else if (key=="D1") ns.d1=val;
                else if (key=="D2") ns.d2=val;
                else if (key=="S1") ns.s1=val;
                else if (key=="S2") ns.s2=val;
                else {std::cout<<"Unknown key '"<<key<<"' (use L1 L2 D1 D2 S1 S2)\n";ok=false;break;}
            }
            if (!ok) continue;
            if (!state_valid(ns)) {
                std::cout << "Invalid state: lives 1–10, doubloons 0–24, shields sum ≤ 30.\n";
            } else {
                game=ns; phase=1;
                std::cout << "State set.\n";
            }
            continue;
        }

        // ── roll DDXJCW ──
        if (cmd=="roll") {
            std::string rest; std::getline(iss,rest);
            DiceCombo combo{};
            if (!parse_dice(rest,combo)) {
                std::cout << "Need exactly 6 dice faces (D/X/J/C/W/S), e.g.: roll D D X J C W\n";
                continue;
            }
            int di=combo_key_map[combo_key(combo)];
            if (di<0){std::cout<<"Bad combo.\n";continue;}

            std::cout << "Rolled: " << combo_str(combo) << "\n";

            if (phase==1 || phase==2) {
                int rerolls = (phase==1) ? 2 : 1;
                AdviceResult ar = advise(game, di, rerolls, V.data());
                const DiceCombo& keep = all_keeps[ar.keep_ki];
                print_keep_advice(combo, keep);
                std::cout << std::fixed << std::setprecision(2)
                          << "  Expected win prob (optimal from here): "
                          << (ar.ev*100.0) << "%\n";
                if (err_bound >= 0.0) {
                    // Keep values inherit the table's error bound, so a margin above
                    // 2*err (plus rounding slack) cannot be overturned by the true values.
                    double gap = ar.ev - ar.runner_up;
                    std::cout << std::scientific << std::setprecision(2);
                    if (gap > 2.0 * err_bound + 1e-13)
                        std::cout << "  Proven optimal: beats the next-best keep by " << gap << "\n";
                    else
                        std::cout << "  Near-tie: next-best keep is within " << gap
                                  << "; this keep is proven within " << 2.0 * err_bound
                                  << " of optimal\n";
                    std::cout << std::defaultfloat;
                }
                phase++;
            } else {
                // Phase 3: score the final dice
                long long nidx = compute_next_idx(combo, game);
                if (nidx==IDX_WIN) {
                    std::cout << (misere ? "  → You reached 25 doubloons or sank the opponent: you LOSE (misère). Game over.\n"
                                         : "  → YOU WIN! Game over.\n");
                    game={9,9,5,5,0,0}; phase=1;
                } else if (nidx==IDX_LOSS) {
                    std::cout << (misere ? "  → Out of lives: YOU WIN (misère)! Game over.\n"
                                         : "  → You lose. Game over.\n");
                    game={9,9,5,5,0,0}; phase=1;
                } else {
                    // nidx is state_index of the next state from opponent's perspective
                    // Decode: (l1=opp_lives-1, l2=my_lives-1, d1=opp_doub, d2=my_doub,
                    //          s1=opp_shields, s2=my_shields)
                    State opp_view = decode_state(nidx);
                    // From MY perspective: l1=opp_view.l2, l2=opp_view.l1, etc.
                    State my_view = {opp_view.l2, opp_view.l1,
                                     opp_view.d2, opp_view.d1,
                                     opp_view.s2, opp_view.s1};
                    double opp_win = V[nidx];
                    double my_win  = 1.0 - opp_win;
                    std::cout << std::fixed << std::setprecision(1);
                    std::cout << "  After your turn:\n"
                              << "    My lives: "    << (my_view.l1+1)
                              << "   My doubloons: " << my_view.d1
                              << "   My shields: "   << my_view.s1 << "\n"
                              << "    Opp lives: "   << (my_view.l2+1)
                              << "  Opp doubloons: " << my_view.d2
                              << "  Opp shields: "   << my_view.s2 << "\n"
                              << "  Your win probability going forward: "
                              << (my_win*100.0) << "%\n"
                              << "  (Opponent's turn — use 'flip' to advise them, "
                                 "then 'flip' again when they finish)\n";
                    // Advance to opponent's turn
                    game  = opp_view;
                    phase = 1;
                }
            }
            continue;
        }

        std::cout << "Unknown command '" << cmd << "'. Type 'help'.\n";
    }

    std::cout << "Goodbye." << std::endl;
    return 0;
}
