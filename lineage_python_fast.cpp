/*
lineage_python_fast.cpp

Accelerated C++ translation of the provided Python lineage simulation.

It preserves the stochastic model and lineage protocol:
  - one lineage is followed until the Python stopping condition is met;
  - division time = birth + tau + Normal(0,1);
  - replication waiting time uses the same time_approx formula;
  - all other reactions use independent exponential waiting times;
  - the event with the smallest waiting time fires;
  - after division, one daughter pole is chosen with probability 1/2;
  - loss[k] = 1.0 / lineage_length.

Acceleration strategy:
  - no NumPy-style hazard arrays are allocated at each SSA step;
  - no reaction-vector arrays are built;
  - events are scanned directly;
  - cluster-cluster fusion scans only occupied cluster sizes.

Compile:
    g++ -O3 -std=c++17 lineage_python_fast.cpp -o lineage_python_fast

Run:
    ./lineage_python_fast

Optional arguments:
    ./lineage_python_fast repeats seed tau n s0

Example:
    ./lineage_python_fast 1000 12345 24 2.9 0.04003065

Outputs:
    Terminal:
        Loss prob <mean> err <std>
        Elapsed time: <seconds>

    File:
        tau24_lineage_summary.csv
*/

#include <bits/stdc++.h>
using namespace std;

struct RNG {
    mt19937_64 gen;
    uniform_real_distribution<double> uni;
    normal_distribution<double> norm01;

    RNG(uint64_t seed) : gen(seed), uni(0.0, 1.0), norm01(0.0, 1.0) {}

    double U() {
        double x = uni(gen);
        if (x <= 0.0) return nextafter(0.0, 1.0);
        if (x >= 1.0) return nextafter(1.0, 0.0);
        return x;
    }

    double N01() {
        return norm01(gen);
    }

    bool coin() {
        return U() < 0.5;
    }
};

struct Params {
    double n      = 2.9;
    double s0     = 0.04003065;
    double tau    = 24.0;
    double k_tr   = 1.49;
    double K      = 15.2 / 1.26;
    double V0     = 1.6;
    double rho    = 0.13;

    double beta   = 0.015;
    double sigma  = 0.08;
    double gamma  = 0.05;

    int x0        = 25;
    int max_size  = 200;

    double mu() const {
        return log(2.0) / tau;
    }
};

struct PoleState {
    vector<long long> y;
    explicit PoleState(int max_size = 200) : y(max_size, 0) {}
};

struct State {
    PoleState pole[2];
    explicit State(int max_size = 200) : pole{PoleState(max_size), PoleState(max_size)} {}
};

struct CellSummary {
    int index;
    int parent;
    double birth;
    double div;
    long long init_total;
    long long final_total;
    int x_records;
};

struct Event {
    enum Type {
        NONE,
        REPLICATION,
        FREE_FREE,
        RECRUITMENT,
        FUSION,
        SHEDDING,
        TRANSFER
    };

    Type type = NONE;
    int pole = -1;
    int i = -1;
    int j = -1;
    double dt = numeric_limits<double>::infinity();
};

static inline long long pole_total_raw(const PoleState& p, const Params& P) {
    long long total = p.y[0];

    for (int i = 2; i < P.max_size; ++i) {
        total += (long long)i * p.y[i];
    }

    return total;
}

static inline long long total_plasmids(const State& s, const Params& P) {
    return pole_total_raw(s.pole[0], P) + pole_total_raw(s.pole[1], P);
}

static inline vector<int> occupied_cluster_sizes(const PoleState& p, const Params& P) {
    vector<int> occ;
    occ.reserve(32);

    for (int i = 2; i < P.max_size; ++i) {
        if (p.y[i] > 0) {
            occ.push_back(i);
        }
    }

    return occ;
}

static inline double time_approx(double s, double xi, double y_tot, double eps, const Params& P) {
    double D = P.k_tr * xi;
    if (D <= 0.0) return numeric_limits<double>::infinity();

    double V = P.V0 * exp(P.mu() * s);
    double C = P.K * y_tot / (P.s0 * 6.022e2 * P.n);
    double one_plus = 1.0 + C / V;

    double a = D / pow(one_plus, P.n);
    double a_prime = (P.mu() * P.n * D * C / V) * pow(one_plus, -P.n - 1.0);

    if (a <= 0.0) return numeric_limits<double>::infinity();

    if (abs(a_prime) < 1e-300) {
        return -log(eps) / a;
    }

    double disc = a * a - 2.0 * a_prime * log(eps);
    if (disc < 0.0 && disc > -1e-12) disc = 0.0;
    if (disc < 0.0) return numeric_limits<double>::infinity();

    double dt = (-a + sqrt(disc)) / a_prime;
    if (dt < 0.0 || !isfinite(dt)) return numeric_limits<double>::infinity();

    return dt;
}

static inline void consider_event(Event& best, Event candidate) {
    if (candidate.dt < best.dt) {
        best = candidate;
    }
}

static inline void consider_constant_rate(
    Event& best,
    RNG& rng,
    double rate,
    Event::Type type,
    int pole,
    int i = -1,
    int j = -1
) {
    if (rate <= 0.0 || !isfinite(rate)) return;

    double eps = rng.U();
    double dt = -log(eps) / rate;

    Event e;
    e.type = type;
    e.pole = pole;
    e.i = i;
    e.j = j;
    e.dt = dt;

    consider_event(best, e);
}

Event draw_next_event(const State& s, double t_y, long long y_tot1, long long y_tot2,
                      long long y_tot, const Params& P, RNG& rng) {
    Event best;

    for (int p = 0; p < 2; ++p) {
        const PoleState& pole = s.pole[p];
        long long y_totp = (p == 0 ? y_tot1 : y_tot2);

        if (y_totp <= 0) continue;

        {
            double eps = rng.U();
            double dt = time_approx(t_y, (double)y_totp, (double)y_tot, eps, P);

            Event e;
            e.type = Event::REPLICATION;
            e.pole = p;
            e.dt = dt;
            consider_event(best, e);
        }

        consider_constant_rate(
            best, rng,
            0.5 * P.beta * pole.y[0] * (pole.y[0] - 1),
            Event::FREE_FREE,
            p
        );

        vector<int> occ = occupied_cluster_sizes(pole, P);

        for (int i : occ) {
            if (i + 1 >= P.max_size) continue;

            consider_constant_rate(
                best, rng,
                P.beta * pole.y[0] * pole.y[i],
                Event::RECRUITMENT,
                p,
                i
            );
        }

        for (size_t a = 0; a < occ.size(); ++a) {
            int i = occ[a];

            for (size_t b = a; b < occ.size(); ++b) {
                int j = occ[b];

                if (i + j >= P.max_size) continue;

                double rate = 0.0;

                if (i == j) {
                    rate = 0.5 * P.gamma * pole.y[i] * (pole.y[i] - 1);
                } else {
                    rate = P.gamma * pole.y[i] * pole.y[j];
                }

                consider_constant_rate(best, rng, rate, Event::FUSION, p, i, j);
            }
        }

        for (int i : occ) {
            consider_constant_rate(
                best, rng,
                P.sigma * pole.y[i] * i,
                Event::SHEDDING,
                p,
                i
            );
        }
    }

    consider_constant_rate(best, rng, P.rho * s.pole[0].y[0], Event::TRANSFER, 0);
    consider_constant_rate(best, rng, P.rho * s.pole[1].y[0], Event::TRANSFER, 1);

    return best;
}

void fire_event(State& s, const Event& e, const Params& P) {
    int p = e.pole;
    int q = 1 - p;

    switch (e.type) {
        case Event::REPLICATION:
            s.pole[p].y[0] += 1;
            break;

        case Event::FREE_FREE:
            if (s.pole[p].y[0] >= 2) {
                s.pole[p].y[0] -= 2;
                s.pole[p].y[2] += 1;
            }
            break;

        case Event::RECRUITMENT:
            if (s.pole[p].y[0] > 0 && e.i >= 2 && e.i + 1 < P.max_size && s.pole[p].y[e.i] > 0) {
                s.pole[p].y[0] -= 1;
                s.pole[p].y[e.i] -= 1;
                s.pole[p].y[e.i + 1] += 1;
            }
            break;

        case Event::FUSION:
            if (e.i >= 2 && e.j >= 2 && e.i + e.j < P.max_size) {
                if (e.i == e.j) {
                    if (s.pole[p].y[e.i] >= 2) {
                        s.pole[p].y[e.i] -= 2;
                        s.pole[p].y[e.i + e.j] += 1;
                    }
                } else {
                    if (s.pole[p].y[e.i] > 0 && s.pole[p].y[e.j] > 0) {
                        s.pole[p].y[e.i] -= 1;
                        s.pole[p].y[e.j] -= 1;
                        s.pole[p].y[e.i + e.j] += 1;
                    }
                }
            }
            break;

        case Event::SHEDDING:
            if (e.i >= 2 && s.pole[p].y[e.i] > 0) {
                s.pole[p].y[e.i] -= 1;

                if (e.i == 2) {
                    s.pole[p].y[0] += 2;
                } else {
                    s.pole[p].y[0] += 1;
                    s.pole[p].y[e.i - 1] += 1;
                }
            }
            break;

        case Event::TRANSFER:
            if (s.pole[p].y[0] > 0) {
                s.pole[p].y[0] -= 1;
                s.pole[q].y[0] += 1;
            }
            break;

        default:
            break;
    }
}

void update_totals(const State& s, const Params& P,
                   long long& y_tot1, long long& y_tot2, long long& y_tot) {
    y_tot1 = pole_total_raw(s.pole[0], P);
    y_tot2 = pole_total_raw(s.pole[1], P);
    y_tot = y_tot1 + y_tot2;
}

struct SSAReturn {
    State final_state;
    vector<long long> x;
    vector<double> t_x;
    vector<State> clust;
    vector<double> ty;

    explicit SSAReturn(int max_size = 200) : final_state(max_size) {}
};

SSAReturn ssa_vol(const State& y0, long long x0, double tau_cell, const Params& P, RNG& rng) {
    SSAReturn out(P.max_size);
    out.final_state = y0;

    State& s = out.final_state;

    long long y_tot1 = x0;
    long long y_tot2 = 0;
    long long y_tot = y_tot1 + y_tot2;

    out.x.push_back(x0);
    out.t_x.push_back(0.0);
    out.clust.push_back(s);

    double t_y = 0.0;

    while (t_y < tau_cell) {
        Event e = draw_next_event(s, t_y, y_tot1, y_tot2, y_tot, P, rng);

        if (e.type == Event::NONE || !isfinite(e.dt)) {
            break;
        }

        fire_event(s, e, P);
        update_totals(s, P, y_tot1, y_tot2, y_tot);

        if (e.type == Event::REPLICATION) {
            out.x.push_back(y_tot);
            out.t_x.push_back(t_y);
        }

        t_y += e.dt;
        out.ty.push_back(t_y);
        out.clust.push_back(s);
    }

    return out;
}

long long initial_total_from_state(const State& s, const Params& P) {
    return pole_total_raw(s.pole[0], P);
}

struct LineageReturn {
    vector<CellSummary> line;
    int length = 1;
    State last_clust;
    vector<double> last_ty;

    explicit LineageReturn(int max_size = 200) : last_clust(max_size) {}
};

LineageReturn lineage_sim(const Params& P, RNG& rng) {
    LineageReturn result(P.max_size);

    int N = 0;
    int length = 1;
    int status = 1;

    State current_init(P.max_size);
    current_init.pole[0].y[0] = P.x0;

    int current_index = 0;
    int current_parent = -1;
    double current_birth = 0.0;
    double current_div = current_birth + P.tau + rng.N01();

    while (status == 1) {
        long long init_total = initial_total_from_state(current_init, P);

        SSAReturn sim(P.max_size);
        if (init_total > 0) {
            sim = ssa_vol(current_init, init_total, current_div - current_birth, P, rng);
        } else {
            sim.final_state = current_init;
            sim.x = {0, 0};
            sim.t_x = {0.0, current_div - current_birth};
            sim.clust = {current_init};
            sim.ty = {0.0, current_div - current_birth};
        }

        long long final_total = total_plasmids(sim.final_state, P);

        CellSummary cs;
        cs.index = current_index;
        cs.parent = current_parent;
        cs.birth = current_birth;
        cs.div = current_div;
        cs.init_total = init_total;
        cs.final_total = final_total;
        cs.x_records = (int)sim.x.size();

        result.line.push_back(cs);
        if (result.line.size() > 20) {
            result.line.erase(result.line.begin());
        }

        result.last_clust = sim.final_state;
        result.last_ty = sim.ty;

        bool keep_first_pole = rng.coin();

        State daughter(P.max_size);
        if (keep_first_pole) {
            daughter.pole[0] = sim.final_state.pole[0];
        } else {
            daughter.pole[0] = sim.final_state.pole[1];
        }

        long long daughter_x0 = pole_total_raw(daughter.pole[0], P);
        long long last_x = sim.x.empty() ? 0 : sim.x.back();

        if (last_x == daughter_x0 || daughter_x0 == 0) {
            status = 0;
        } else {
            length += 1;
            status = 1;

            current_init = daughter;
            current_parent = current_index;
            current_index = N + 1;
            current_birth = current_div;
            current_div = current_birth + P.tau + rng.N01();

            N += 1;
        }
    }

    result.length = length;
    return result;
}

void write_lineage_summary_csv(const string& filename, const vector<CellSummary>& line) {
    ofstream out(filename);
    out << "index,parent,birth,division,init_total,final_total,x_records\n";

    for (const auto& c : line) {
        out << c.index << ","
            << c.parent << ","
            << setprecision(17) << c.birth << ","
            << setprecision(17) << c.div << ","
            << c.init_total << ","
            << c.final_total << ","
            << c.x_records << "\n";
    }
}

int main(int argc, char* argv[]) {
    Params P;

    int repeats = 1;
    uint64_t seed = chrono::high_resolution_clock::now().time_since_epoch().count();

    if (argc >= 2) repeats = stoi(argv[1]);
    if (argc >= 3) seed = stoull(argv[2]);
    if (argc >= 4) P.tau = stod(argv[3]);
    if (argc >= 5) P.n = stod(argv[4]);
    if (argc >= 6) P.s0 = stod(argv[5]);

    RNG rng(seed);

    vector<double> loss(repeats, 0.0);
    vector<CellSummary> last_line;

    auto start = chrono::steady_clock::now();

    for (int k = 0; k < repeats; ++k) {
        LineageReturn lr = lineage_sim(P, rng);
        loss[k] = 1.0 / (double)lr.length;
        last_line = lr.line;
    }

    double mean = 0.0;
    for (double v : loss) mean += v;
    mean /= (double)repeats;

    double var = 0.0;
    for (double v : loss) {
        double d = v - mean;
        var += d * d;
    }
    var /= (double)repeats;

    double sd = sqrt(var);

    auto end = chrono::steady_clock::now();
    double elapsed = chrono::duration<double>(end - start).count();

    cout << "Loss prob " << setprecision(12) << mean
         << " err " << setprecision(12) << sd << "\n";
    cout << "Elapsed time:  " << setprecision(12) << elapsed << "\n";

    string tau_str;
    {
        ostringstream ss;
        ss << "tau" << (long long)llround(P.tau);
        tau_str = ss.str();
    }

    write_lineage_summary_csv(tau_str + "_lineage_summary.csv", last_line);

    return 0;
}

