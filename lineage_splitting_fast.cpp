
/*
lineage_splitting_fast.cpp

Weighted splitting / RESTART-style rare-event simulation for the plasmid
lineage model.

Purpose
-------
This estimates the same quantity as the previous direct lineage code:

    direct lineage estimate = 1 / lineage_length

where lineage_length is the number of cell cycles until the Python stopping
criterion is met.

This code uses weighted splitting:
  - when a lineage enters a lower-copy-number region, it is cloned;
  - each clone receives a smaller weight;
  - terminal descendants contribute weight * (1 / lineage_length);
  - weights preserve the same estimator as direct simulation.

Thus the event rules are unchanged. Splitting only changes the sampling.

Compile
-------
    g++ -O3 -std=c++17 lineage_splitting_fast.cpp -o lineage_splitting_fast

Run
---
Default:
    ./lineage_splitting_fast

Recommended test:
    ./lineage_splitting_fast 1 12345 10 4

Larger run:
    ./lineage_splitting_fast 1 12345 100 4

Arguments:
    ./lineage_splitting_fast repeats seed initial_particles split_factor tau n s0

Example:
    ./lineage_splitting_fast 5 12345 100 4 24 2.9 0.04003065

Output
------
Prints:
    Loss prob <mean> err <sd>
    Elapsed time: <seconds>

Also writes:
    splitting_results.csv
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

struct CellCycleResult {
    State final_state;
    long long last_x;
    long long final_total;
    uint64_t ssa_steps;

    explicit CellCycleResult(int max_size = 200)
        : final_state(max_size), last_x(0), final_total(0), ssa_steps(0) {}
};

struct Particle {
    State state;
    double weight;
    int level;
    long long age_cycles;

    explicit Particle(int max_size = 200)
        : state(max_size), weight(1.0), level(0), age_cycles(0) {}
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

static inline void consider_event(Event& best, const Event& candidate) {
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

    Event e;
    e.type = type;
    e.pole = pole;
    e.i = i;
    e.j = j;
    e.dt = -log(rng.U()) / rate;

    consider_event(best, e);
}

Event draw_next_event(
    const State& s,
    double t_y,
    long long y_tot1,
    long long y_tot2,
    long long y_tot,
    const Params& P,
    RNG& rng
) {
    Event best;

    for (int p = 0; p < 2; ++p) {
        const PoleState& pole = s.pole[p];
        long long y_totp = (p == 0 ? y_tot1 : y_tot2);

        if (y_totp <= 0) continue;

        {
            Event e;
            e.type = Event::REPLICATION;
            e.pole = p;
            e.dt = time_approx(t_y, (double)y_totp, (double)y_tot, rng.U(), P);
            consider_event(best, e);
        }

        consider_constant_rate(
            best,
            rng,
            0.5 * P.beta * pole.y[0] * (pole.y[0] - 1),
            Event::FREE_FREE,
            p
        );

        vector<int> occ = occupied_cluster_sizes(pole, P);

        for (int i : occ) {
            if (i + 1 >= P.max_size) continue;

            consider_constant_rate(
                best,
                rng,
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
                best,
                rng,
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
            if (s.pole[p].y[0] > 0 &&
                e.i >= 2 &&
                e.i + 1 < P.max_size &&
                s.pole[p].y[e.i] > 0) {
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
                   long long& y_tot1,
                   long long& y_tot2,
                   long long& y_tot) {
    y_tot1 = pole_total_raw(s.pole[0], P);
    y_tot2 = pole_total_raw(s.pole[1], P);
    y_tot = y_tot1 + y_tot2;
}

CellCycleResult simulate_cell_cycle(
    const State& y0,
    double tau_cell,
    const Params& P,
    RNG& rng
) {
    CellCycleResult out(P.max_size);
    out.final_state = y0;

    State& s = out.final_state;

    long long y_tot1 = pole_total_raw(s.pole[0], P);
    long long y_tot2 = 0;
    long long y_tot = y_tot1 + y_tot2;

    out.last_x = y_tot;

    double t_y = 0.0;

    while (t_y < tau_cell) {
        Event e = draw_next_event(s, t_y, y_tot1, y_tot2, y_tot, P, rng);

        if (e.type == Event::NONE || !isfinite(e.dt)) {
            break;
        }

        // This follows the Python convention: fire the next event, then advance
        // time. Thus an event can be recorded just beyond the division time.
        fire_event(s, e, P);
        out.ssa_steps++;

        update_totals(s, P, y_tot1, y_tot2, y_tot);

        if (e.type == Event::REPLICATION) {
            out.last_x = y_tot;
        }

        t_y += e.dt;
    }

    out.final_total = y_tot;
    return out;
}

int splitting_level(long long inherited_copy_number, const vector<int>& thresholds) {
    int level = 0;

    for (int th : thresholds) {
        if (inherited_copy_number <= th) {
            level++;
        }
    }

    return level;
}

struct SplitRunResult {
    double estimate = 0.0;
    double terminal_weight = 0.0;
    long long terminal_particles = 0;
    long long processed_particles = 0;
    uint64_t total_ssa_steps = 0;
    double max_weight = 0.0;
    size_t max_active_particles = 0;
};

SplitRunResult run_one_splitting_experiment(
    const Params& P,
    RNG& rng,
    int initial_particles,
    int split_factor,
    const vector<int>& thresholds,
    bool verbose
) {
    deque<Particle> q;

    double initial_weight = 1.0 / (double)initial_particles;

    for (int k = 0; k < initial_particles; ++k) {
        Particle p(P.max_size);
        p.state.pole[0].y[0] = P.x0;
        p.weight = initial_weight;
        p.level = splitting_level(P.x0, thresholds);
        p.age_cycles = 0;
        q.push_back(std::move(p));
    }

    SplitRunResult result;
    auto start = chrono::steady_clock::now();

    while (!q.empty()) {
        Particle p = std::move(q.front());
        q.pop_front();

        double tau_cell = P.tau + rng.N01();
        if (tau_cell <= 0.0) {
            // Practically impossible for tau=24, sd=1.
            // Guard only prevents invalid negative cell-cycle duration.
            tau_cell = P.tau;
        }

        CellCycleResult cc = simulate_cell_cycle(p.state, tau_cell, P, rng);
        result.total_ssa_steps += cc.ssa_steps;

        p.age_cycles += 1;
        result.processed_particles += 1;

        bool keep_first_pole = rng.coin();

        State daughter(P.max_size);
        if (keep_first_pole) {
            daughter.pole[0] = cc.final_state.pole[0];
        } else {
            daughter.pole[0] = cc.final_state.pole[1];
        }

        long long daughter_x0 = pole_total_raw(daughter.pole[0], P);

        bool terminal =
            (daughter_x0 == 0) ||
            (cc.last_x == daughter_x0);

        if (terminal) {
            result.terminal_particles += 1;
            result.terminal_weight += p.weight;
            result.estimate += p.weight / (double)p.age_cycles;
        } else {
            int new_level = splitting_level(daughter_x0, thresholds);

            p.state = std::move(daughter);

            if (new_level > p.level && split_factor > 1) {
                double child_weight = p.weight / (double)split_factor;

                for (int m = 0; m < split_factor; ++m) {
                    Particle child = p;
                    child.weight = child_weight;
                    child.level = new_level;
                    q.push_back(std::move(child));
                }
            } else {
                p.level = max(p.level, new_level);
                q.push_back(std::move(p));
            }
        }

        result.max_active_particles = max(result.max_active_particles, q.size());
        result.max_weight = max(result.max_weight, p.weight);

        if (verbose && result.processed_particles % 10000 == 0) {
            auto now = chrono::steady_clock::now();
            double elapsed = chrono::duration<double>(now - start).count();

            cerr << "processed=" << result.processed_particles
                 << " active=" << q.size()
                 << " terminals=" << result.terminal_particles
                 << " estimate_so_far=" << setprecision(12) << result.estimate
                 << " ssa_steps=" << result.total_ssa_steps
                 << " elapsed=" << elapsed << " s\n";
        }
    }

    return result;
}

int main(int argc, char* argv[]) {
    Params P;

    int repeats = 1;
    uint64_t seed = 12345;
    int initial_particles = 50;
    int split_factor = 4;

    if (argc >= 2) repeats = stoi(argv[1]);
    if (argc >= 3) seed = stoull(argv[2]);
    if (argc >= 4) initial_particles = stoi(argv[3]);
    if (argc >= 5) split_factor = stoi(argv[4]);
    if (argc >= 6) P.tau = stod(argv[5]);
    if (argc >= 7) P.n = stod(argv[6]);
    if (argc >= 8) P.s0 = stod(argv[7]);

    if (repeats <= 0 || initial_particles <= 0 || split_factor <= 0) {
        cerr << "Arguments must satisfy repeats>0, initial_particles>0, split_factor>0.\n";
        return 1;
    }

    vector<int> thresholds = {20, 15, 10, 7, 5, 3, 1};

    RNG rng(seed);

    vector<double> estimates;
    estimates.reserve(repeats);

    ofstream csv("splitting_results.csv");
    csv << "repeat,estimate,terminal_weight,terminal_particles,processed_particles,total_ssa_steps,max_active_particles\n";

    auto start = chrono::steady_clock::now();

    for (int r = 0; r < repeats; ++r) {
        bool verbose = (repeats == 1);

        SplitRunResult res = run_one_splitting_experiment(
            P,
            rng,
            initial_particles,
            split_factor,
            thresholds,
            verbose
        );

        estimates.push_back(res.estimate);

        csv << r << ","
            << setprecision(17) << res.estimate << ","
            << setprecision(17) << res.terminal_weight << ","
            << res.terminal_particles << ","
            << res.processed_particles << ","
            << res.total_ssa_steps << ","
            << res.max_active_particles << "\n";

        cerr << "repeat " << r
             << " estimate=" << setprecision(12) << res.estimate
             << " terminal_weight=" << res.terminal_weight
             << " terminal_particles=" << res.terminal_particles
             << " processed_particles=" << res.processed_particles
             << " max_active=" << res.max_active_particles << "\n";
    }

    double mean = 0.0;
    for (double x : estimates) mean += x;
    mean /= (double)estimates.size();

    double var = 0.0;
    for (double x : estimates) {
        double d = x - mean;
        var += d * d;
    }
    var /= (double)estimates.size();

    double sd = sqrt(var);

    auto end = chrono::steady_clock::now();
    double elapsed = chrono::duration<double>(end - start).count();

    cout << "Loss prob " << setprecision(12) << mean
         << " err " << setprecision(12) << sd << "\n";
    cout << "Elapsed time:  " << setprecision(12) << elapsed << "\n";
    cout << "Wrote splitting_results.csv\n";

    return 0;
}

