/**
 * Booty Dice: simulate optimal-vs-optimal games.
 * Usage: ./sim [bin_path] [num_games] [seed]
 */

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

// ── Constants (must match solver) ────────────────────────────────────────────

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
static const char FACE_CHAR[] = "DXJCWS";

// ── State + Dice types ────────────────────────────────────────────────────────

struct State     { int l1,l2,d1,d2,s1,s2; };
struct DiceCombo { int c[NUM_FACES]; };

inline long long state_index(const State& s) {
    return ((((((long long)s.l1*DIM_L+s.l2)*DIM_D+s.d1)*DIM_D+s.d2)*DIM_S+s.s1)*DIM_S+s.s2);
}
inline bool state_valid(const State& s) {
    return s.l1>=0&&s.l1<DIM_L&&s.l2>=0&&s.l2<DIM_L&&
           s.d1>=0&&s.d1<DIM_D&&s.d2>=0&&s.d2<DIM_D&&
           s.s1>=0&&s.s2>=0&&s.s1+s.s2<=MAX_SHIELDS;
}
inline int combo_key(const DiceCombo& c) {
    return ((((c.c[0]*7+c.c[1])*7+c.c[2])*7+c.c[3])*7+c.c[4])*7+c.c[5];
}

// ── Dice tables ───────────────────────────────────────────────────────────────

static std::vector<DiceCombo> all_combos;
static std::vector<DiceCombo> all_keeps;
static int combo_key_map[7*7*7*7*7*7];
static int keep_key_map [7*7*7*7*7*7];

struct SparseEntry { int di; float prob; };
static std::vector<std::vector<SparseEntry>> T_sparse;
static std::vector<std::vector<int>>         valid_keeps;

void enumerate_multisets(int face, int budget, DiceCombo& cur,
                         bool exact, std::vector<DiceCombo>& out) {
    if (face==NUM_FACES){if(!exact||budget==0)out.push_back(cur);return;}
    for(int k=0;k<=budget;++k){cur.c[face]=k;enumerate_multisets(face+1,budget-k,cur,exact,out);}
    cur.c[face]=0;
}

void build_dice_tables() {
    memset(combo_key_map,-1,sizeof(combo_key_map));
    memset(keep_key_map, -1,sizeof(keep_key_map));
    DiceCombo tmp{};
    enumerate_multisets(0,NUM_DICE,tmp,true, all_combos);
    enumerate_multisets(0,NUM_DICE,tmp,false,all_keeps);
    for(int i=0;i<(int)all_combos.size();++i) combo_key_map[combo_key(all_combos[i])]=i;
    for(int i=0;i<(int)all_keeps.size(); ++i) keep_key_map [combo_key(all_keeps[i])] =i;

    int nc=(int)all_combos.size(), nk=(int)all_keeps.size();
    std::vector<std::vector<float>> T(nk,std::vector<float>(nc,0.0f));
    for(int ki=0;ki<nk;++ki){
        const DiceCombo& keep=all_keeps[ki];
        int kept=0; for(int f=0;f<NUM_FACES;++f) kept+=keep.c[f];
        int reroll=NUM_DICE-kept;
        long long total=1; for(int i=0;i<reroll;++i) total*=NUM_FACES;
        for(long long out2=0;out2<total;++out2){
            DiceCombo res=keep; long long t=out2;
            for(int r=0;r<reroll;++r){res.c[t%NUM_FACES]++;t/=NUM_FACES;}
            int ci=combo_key_map[combo_key(res)];
            if(ci>=0) T[ki][ci]+=1.0f;
        }
        float denom=(float)total;
        for(int ci=0;ci<nc;++ci) T[ki][ci]/=denom;
    }
    T_sparse.resize(nk);
    for(int ki=0;ki<nk;++ki)
        for(int ci=0;ci<nc;++ci)
            if(T[ki][ci]>0.0f) T_sparse[ki].push_back({ci,T[ki][ci]});
    valid_keeps.resize(nc);
    for(int ci=0;ci<nc;++ci){
        const DiceCombo& combo=all_combos[ci];
        for(int ki=0;ki<nk;++ki){
            const DiceCombo& keep=all_keeps[ki];
            bool ok=true;
            for(int f=0;f<NUM_FACES;++f) if(keep.c[f]>combo.c[f]){ok=false;break;}
            if(ok) valid_keeps[ci].push_back(ki);
        }
    }
}

// ── Dice resolution ───────────────────────────────────────────────────────────

inline long long compute_next_idx(const DiceCombo& combo, const State& s) {
    int cD=combo.c[F_D],cX=combo.c[F_X],cJ=combo.c[F_J];
    int cC=combo.c[F_C],cW=combo.c[F_W],cS=combo.c[F_S];
    int ml=s.l1+1,ol=s.l2+1,md=s.d1,od=s.d2,ms=s.s1,os=s.s2;
    int wc=MAX_SHIELDS-ms-os;
    if(cD==1&&cX==1&&cJ==1&&cC==1&&cW==1&&cS==1){
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
    ml=std::max(ml,0);ol=std::max(ol,0);md=std::max(md,0);od=std::max(od,0);
    if(md>=WIN_DOUBLOONS||ol<=0) return IDX_WIN;
    if(ml<=0)                    return IDX_LOSS;
    State next{ol-1,ml-1,od,md,os,ms};
    return state_index(next);
}

long long resolve_verbose(const DiceCombo& combo, const State& s, std::ostream& out) {
    int cD=combo.c[F_D],cX=combo.c[F_X],cJ=combo.c[F_J];
    int cC=combo.c[F_C],cW=combo.c[F_W],cS=combo.c[F_S];
    int ml=s.l1+1,ol=s.l2+1,md=s.d1,od=s.d2,ms=s.s1,os=s.s2;
    int wc=MAX_SHIELDS-ms-os;

    if(cD==1&&cX==1&&cJ==1&&cC==1&&cW==1&&cS==1){
        int give=od<5?od:5;
        out<<"  [Blackbeard's Curse] Opponent loses 2 lives; gives "<<give<<" doubloons to Buried Treasure.\n";
        ol-=2; od-=give;
    } else {
        // Doubloon
        if(cD>0){
            out<<"  [Doubloon D="<<cD<<"] You gain "<<2*cD<<" doubloons.";
            md+=2*cD;
            if(md>=WIN_DOUBLOONS){
                out<<" -> You reach "<<md<<" doubloons -- YOU LOSE MISERE!\n";
                return IDX_WIN;
            }
            out<<"\n";
        }
        // X Marks the Spot (not Shipwreck)
        if(cX>0&&cX<3){
            int lose=md<2*cX?md:2*cX;
            out<<"  [X Marks the Spot X="<<cX<<"] You lose "<<lose<<" doubloons.\n";
            md-=lose;
        }
        // Jolly Roger
        if(cJ>0){
            int steal=od<2*cJ?od:2*cJ;
            out<<"  [Jolly Roger J="<<cJ<<"] You steal "<<steal<<" doubloon"<<(steal!=1?"s":"")<<" from opponent.\n";
            md+=steal; od-=steal;
        }
        // Cutlass — immediate win if opponent dies
        if(cC>0){
            int ab=os<cC?os:cC;
            int lives_lost=cC-ab;
            if(ab>0) out<<"  [Cutlass C="<<cC<<"] Blocked "<<ab<<" by opponent shields";
            else     out<<"  [Cutlass C="<<cC<<"]";
            if(lives_lost>0) out<<" -> opponent loses "<<lives_lost<<(lives_lost>1?" lives":" life");
            out<<".\n";
            os-=ab; wc+=ab; ol-=lives_lost;
            if(ol<=0){
                out<<"  -> Opponent is out of lives -- YOU LOSE MISERE!\n";
                return IDX_WIN;
            }
        }
        // Walk the Plank (not Mutiny)
        if(cW>0&&cW<3){
            out<<"  [Walk the Plank W="<<cW<<"] You lose "<<cW<<(cW>1?" lives":" life")<<".\n";
            ml-=cW;
        }
        // Shield
        if(cS>0){
            int take=wc<cS?wc:cS;
            out<<"  [Shield S="<<cS<<"] You gain "<<take<<(take!=1?" shields":" shield")
               <<" (war chest had "<<wc<<").\n";
            ms+=take;
        }
        // Mutiny
        if(cW>=3){
            out<<"  [Mutiny W="<<cW<<">=3] Opponent loses "<<(cW-2)<<(cW-2>1?" lives":" life")<<".\n";
            ol-=(cW-2);
        }
        // Shipwreck (deferred past Jolly Roger)
        if(cX>=3){
            int give=od<cX?od:cX;
            out<<"  [Shipwreck X="<<cX<<">=3] Opponent gives "<<give<<" doubloon"<<(give!=1?"s":"")<<" to Buried Treasure.\n";
            od-=give;
        }
    }

    ml=std::max(ml,0);ol=std::max(ol,0);md=std::max(md,0);od=std::max(od,0);
    if(md>=WIN_DOUBLOONS){out<<"  -> You reach "<<md<<" doubloons -- YOU LOSE MISERE!\n"; return IDX_WIN;}
    if(ol<=0)            {out<<"  -> Opponent is out of lives -- YOU LOSE MISERE!\n";      return IDX_WIN;}
    if(ml<=0)            {out<<"  -> You are out of lives -- YOU WIN MISERE!\n";           return IDX_LOSS;}
    State next{ol-1,ml-1,od,md,os,ms};
    return state_index(next);
}

// ── Advisor ───────────────────────────────────────────────────────────────────

int best_keep(const State& s, int combo_di, int rerolls, const float* V) {
    int nc=(int)all_combos.size(), nk=(int)all_keeps.size();
    std::vector<float> end_val(nc), keep_ev(nk), roll2_ev(nc);

    for(int di=0;di<nc;++di){
        long long n=compute_next_idx(all_combos[di],s);
        if(n==IDX_WIN)  end_val[di]=0.0f;   // misère: reaching 25D or killing opp is BAD
        else if(n==IDX_LOSS) end_val[di]=1.0f;  // misère: dying is GOOD
        else            end_val[di]=1.0f-V[n];
    }
    for(int ki=0;ki<nk;++ki){
        float acc=0.0f;
        for(const auto& e:T_sparse[ki]) acc+=e.prob*end_val[e.di];
        keep_ev[ki]=acc;
    }
    if(rerolls==1){
        float best=-1.0f; int bki=-1;
        for(int ki:valid_keeps[combo_di]) if(keep_ev[ki]>best){best=keep_ev[ki];bki=ki;}
        return bki;
    }
    for(int di=0;di<nc;++di){
        if(end_val[di]==1.0f){roll2_ev[di]=1.0f;continue;}
        float best=0.0f;
        for(int ki:valid_keeps[di]) if(keep_ev[ki]>best) best=keep_ev[ki];
        roll2_ev[di]=best;
    }
    std::vector<float> keep_ev1(nk);
    for(int ki=0;ki<nk;++ki){
        float acc=0.0f;
        for(const auto& e:T_sparse[ki]) acc+=e.prob*roll2_ev[e.di];
        keep_ev1[ki]=acc;
    }
    float best=-1.0f; int bki=-1;
    for(int ki:valid_keeps[combo_di]) if(keep_ev1[ki]>best){best=keep_ev1[ki];bki=ki;}
    return bki;
}

// ── Formatting ────────────────────────────────────────────────────────────────

std::string combo_str(const DiceCombo& c) {
    std::string s;
    for(int f=0;f<NUM_FACES;++f)
        for(int k=0;k<c.c[f];++k){ s+=FACE_CHAR[f]; s+=' '; }
    if(!s.empty()) s.pop_back();
    return s;
}

std::string keep_str(const DiceCombo& roll, const DiceCombo& keep) {
    std::string sk,sd;
    for(int f=0;f<NUM_FACES;++f){
        for(int k=0;k<keep.c[f];++k)        {sk+=FACE_CHAR[f];sk+=' ';}
        for(int k=0;k<roll.c[f]-keep.c[f];++k){sd+=FACE_CHAR[f];sd+=' ';}
    }
    if(!sk.empty()) sk.pop_back();
    if(!sd.empty()) sd.pop_back();
    return "keep [" + (sk.empty()?"(none)":sk) + "]" +
           (sd.empty() ? " (no reroll)" : "  reroll [" + sd + "]");
}

// ── Random rolling ────────────────────────────────────────────────────────────

DiceCombo roll_dice(std::mt19937& rng, int n) {
    DiceCombo c{};
    std::uniform_int_distribution<int> d6(0,NUM_FACES-1);
    for(int i=0;i<n;++i) c.c[d6(rng)]++;
    return c;
}

DiceCombo apply_reroll(std::mt19937& rng, const DiceCombo& keep) {
    int kept=0; for(int f=0;f<NUM_FACES;++f) kept+=keep.c[f];
    DiceCombo extra = roll_dice(rng, NUM_DICE-kept);
    DiceCombo result = keep;
    for(int f=0;f<NUM_FACES;++f) result.c[f]+=extra.c[f];
    return result;
}

// ── Game simulation ───────────────────────────────────────────────────────────

// Returns 1 if P1 wins, 2 if P2 wins.
// State is always from current player's perspective (P1 starts).
int simulate_game(std::mt19937& rng, const float* V, int game_num, std::ostream& out) {
    out << "\n==========================================\n";
    out << "  GAME " << game_num << " (MISERE -- trying to lose)\n";
    out << "==========================================\n";

    State state = {9,9,5,5,0,0};
    int cur_player = 1;
    int turn = 1;

    for(;;){
        float win_prob = state_valid(state) ? V[state_index(state)] : 0.5f;
        out << "\n-- Turn " << turn << "  P" << cur_player
            << " to roll ----------------------------------\n";
        out << "  State: L" << (state.l1+1) << "/" << (state.l2+1)
            << "  D" << state.d1 << "/" << state.d2
            << "  S" << state.s1 << "/" << state.s2
            << "  (P" << cur_player << " misere-win prob: "
            << std::fixed << std::setprecision(1) << (win_prob*100.0f) << "%)\n";

        DiceCombo roll1 = roll_dice(rng, NUM_DICE);
        out << "  Roll 1: " << combo_str(roll1) << "\n";
        int di1 = combo_key_map[combo_key(roll1)];
        int ki1 = best_keep(state, di1, 2, V);
        const DiceCombo& keep1 = all_keeps[ki1];
        out << "  -> " << keep_str(roll1, keep1) << "\n";

        DiceCombo roll2 = apply_reroll(rng, keep1);
        out << "  Roll 2: " << combo_str(roll2) << "\n";
        int di2 = combo_key_map[combo_key(roll2)];
        int ki2 = best_keep(state, di2, 1, V);
        const DiceCombo& keep2 = all_keeps[ki2];
        out << "  -> " << keep_str(roll2, keep2) << "\n";

        DiceCombo roll3 = apply_reroll(rng, keep2);
        out << "  Roll 3: " << combo_str(roll3) << "  (final)\n";
        out << "  Resolution:\n";
        long long nidx = resolve_verbose(roll3, state, out);

        if(nidx == IDX_WIN || nidx == IDX_LOSS){
            // misère: IDX_WIN (normal-game win) = cur_player loses; IDX_LOSS = cur_player wins
            int winner = (nidx==IDX_LOSS) ? cur_player : (3-cur_player);
            out << "\n  *** P" << winner << " wins the misere game! (in " << turn << " turns) ***\n";
            return winner;
        }

        State& ns = state;
        {
            long long idx = nidx;
            int s2=idx%DIM_S;idx/=DIM_S;
            int s1=idx%DIM_S;idx/=DIM_S;
            int d2=idx%DIM_D;idx/=DIM_D;
            int d1=idx%DIM_D;idx/=DIM_D;
            int l2=idx%DIM_L;idx/=DIM_L;
            int l1=(int)idx;
            ns={l1,l2,d1,d2,s1,s2};
        }

        out << "  After: P" << cur_player
            << " L" << (state.l2+1) << " D" << state.d2 << " S" << state.s2
            << "  |  P" << (3-cur_player)
            << " L" << (state.l1+1) << " D" << state.d1 << " S" << state.s1 << "\n";

        cur_player = 3 - cur_player;
        ++turn;

        if(turn > 500){
            out << "  (game truncated at 500 turns)\n";
            return 0;
        }
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    const char* bin_path  = (argc>1) ? argv[1] : "booty_dice_misere.bin";
    int  num_games        = (argc>2) ? std::stoi(argv[2]) : 3;
    unsigned seed         = (argc>3) ? (unsigned)std::stoul(argv[3]) : 42u;

    std::cout << "Building dice tables... " << std::flush;
    build_dice_tables();
    std::cout << "done.\n";

    std::cout << "Loading " << bin_path << "... " << std::flush;
    std::vector<float> V(TOTAL_STATES);
    {
        std::ifstream in(bin_path, std::ios::binary);
        if(!in){ std::cerr<<"Cannot open '"<<bin_path<<"'\n"; return 1; }
        uint32_t magic=0, version=0; int dims[6]={};
        in.read(reinterpret_cast<char*>(&magic),  4);
        in.read(reinterpret_cast<char*>(&version),4);
        in.read(reinterpret_cast<char*>(dims),    sizeof(dims));
        if(version==2){
            // float64 table from ./verify --save-f64: skip variant + error bound
            char extra[16]; in.read(extra,sizeof(extra));
            std::vector<double> Vd(TOTAL_STATES);
            in.read(reinterpret_cast<char*>(Vd.data()),TOTAL_STATES*sizeof(double));
            std::copy(Vd.begin(),Vd.end(),V.begin());
        } else {
            in.read(reinterpret_cast<char*>(V.data()),TOTAL_STATES*sizeof(float));
        }
        if(!in){ std::cerr<<"Truncated file.\n"; return 1; }
    }
    std::cout << "done.\n";

    std::mt19937 rng(seed);
    int p1_wins=0, p2_wins=0;
    for(int g=1; g<=num_games; ++g){
        std::ostringstream fname;
        fname << "game_" << std::setw(3) << std::setfill('0') << g << ".txt";
        std::ofstream fout(fname.str());
        if(!fout){ std::cerr<<"Cannot open "<<fname.str()<<"\n"; return 1; }

        int w = simulate_game(rng, V.data(), g, fout);
        fout.close();

        if(w==1) ++p1_wins;
        else if(w==2) ++p2_wins;
        std::cout << "  Game " << g << " -> P" << w << " wins  [" << fname.str() << "]\n";
    }

    std::cout << "\nResults over " << num_games << " games:\n";
    std::cout << "  P1 (first mover) wins: " << p1_wins << "\n";
    std::cout << "  P2 wins:               " << p2_wins << "\n";
    return 0;
}
