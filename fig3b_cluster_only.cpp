// fig3b_cluster_only.cpp
//
// Fast exact SSA / hybrid-direct simulation for Fig. 3B
// (clustering model only).
//
// Uses:
//   n      = 2.56
//   s0     = 0.034
//   beta   = 0.015
//   sigma  = 0.08
//   gamma  = 0.05
//   rho    = 0.13
//
// Generation times:
//   24, 30, 40 min
//
// Output:
//   One line per lineage:
//
//     tau generations_to_loss
//
// Example:
//     24 18342
//     30 991203
//
// Designed for embarrassingly parallel runs.
//
// --------------------------------------------------------------------

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

static constexpr double INF =
    std::numeric_limits<double>::infinity();

struct Params {

    // replication
    double n_hill = 2.56;
    double s0     = 0.034;
    double k_tr   = 2.17;
    double K      = 15.8 / 1.26;
    double V0     = 1.2;

    // clustering
    double rho    = 0.13;
    double beta   = 0.015;
    double sigma  = 0.08;
    double gamma  = 0.05;

    // simulation
    int max_size  = 60;
    int x0        = 25;

    // growth
    double tau;
    double mu;
};

using Pole = std::vector<int>;

static inline int pole_total(const Pole& y) {
    int s = y[0];
    for (int k = 2; k < (int)y.size(); ++k)
        s += k * y[k];
    return s;
}

// ------------------------------------------------------------

static inline double time_approx(
    double s,
    int xi,
    int y_tot,
    double eps,
    const Params& p)
{
    if (xi <= 0 || eps <= 0.0)
        return INF;

    const double D = p.k_tr * xi;

    const double V =
        p.V0 * std::exp(p.mu * s);

    const double C =
        p.K * y_tot /
        (p.s0 * 602.2 * p.n_hill);

    const double ratio = C / V;

    const double fn =
        std::pow(1.0 + ratio, p.n_hill);

    const double a = D / fn;

    if (a <= 0.0)
        return INF;

    const double a_prime =
        (p.mu * p.n_hill * D * ratio)
        / ((1.0 + ratio) * fn);

    const double log_eps = std::log(eps);

    if (std::fabs(a_prime) < 1e-14 * a)
        return -log_eps / a;

    const double disc =
        a * a - 2.0 * a_prime * log_eps;

    if (disc < 0.0)
        return INF;

    const double dt =
        (-a + std::sqrt(disc)) / a_prime;

    return (dt > 0.0) ? dt : INF;
}

// ------------------------------------------------------------

static void ssa_cycle(
    Pole& y1,
    Pole& y2,
    const Params& p,
    std::mt19937_64& rng)
{
    const int ms = p.max_size;

    std::uniform_real_distribution<double> U(0.0,1.0);

    struct Rxn {
        double a;
        int type;
        int pole;
        int i;
        int j;
    };

    static thread_local std::vector<Rxn> rxns;
    static thread_local std::vector<int> act1, act2;

    rxns.reserve(256);
    act1.reserve(32);
    act2.reserve(32);

    int ytot1 = pole_total(y1);
    int ytot2 = pole_total(y2);
    int ytot  = ytot1 + ytot2;

    act1.clear();
    act2.clear();

    for (int k=2; k<ms; ++k) {
        if (y1[k] > 0) act1.push_back(k);
        if (y2[k] > 0) act2.push_back(k);
    }

    auto act_add =
    [](std::vector<int>& act, int k)
    {
        auto it =
            std::lower_bound(act.begin(),
                             act.end(), k);

        if (it == act.end() || *it != k)
            act.insert(it,k);
    };

    auto act_rem =
    [](std::vector<int>& act, int k)
    {
        auto it =
            std::lower_bound(act.begin(),
                             act.end(), k);

        if (it != act.end() && *it == k)
            act.erase(it);
    };

    auto upd =
    [&](Pole& y,
        std::vector<int>& act,
        int k,
        int delta)
    {
        const int old = y[k];

        y[k] += delta;

        if (old == 0 && y[k] > 0)
            act_add(act,k);

        else if (old > 0 && y[k] == 0)
            act_rem(act,k);
    };

    double t = 0.0;

    while (t < p.tau) {

        if (ytot == 0)
            break;

        rxns.clear();

        double A0 = 0.0;

        auto add =
        [&](double rate,
            int type,
            int pole,
            int i,
            int j)
        {
            if (rate <= 0.0)
                return;

            rxns.push_back(
                {rate,type,pole,i,j});

            A0 += rate;
        };

        for (int pole=0; pole<2; ++pole) {

            const Pole& y =
                (pole==0)? y1 : y2;

            const std::vector<int>& act =
                (pole==0)? act1 : act2;

            const int M = (int)act.size();
            const int f = y[0];

            if (f >= 2)
                add(
                    0.5 * p.beta * f * (f-1),
                    1,pole,0,0);

            for (int k : act)
                if (k+1 < ms)
                    add(
                        p.beta * f * y[k],
                        2,pole,k,0);

            for (int ai=0; ai<M; ++ai) {

                const int i  = act[ai];
                const int ci = y[i];

                for (int aj=ai; aj<M; ++aj) {

                    const int j = act[aj];

                    if (i+j >= ms)
                        continue;

                    const double rate =
                        (i==j)
                        ?
                        0.5 * p.gamma
                        * ci * (ci-1)
                        :
                        p.gamma
                        * ci * y[j];

                    add(rate,3,pole,i,j);
                }
            }

            for (int k : act)
                add(
                    p.sigma * k * y[k],
                    4,pole,k,0);
        }

        if (y1[0] > 0)
            add(p.rho*y1[0],5,-1,0,0);

        if (y2[0] > 0)
            add(p.rho*y2[0],6,-1,0,0);

        const double t_hom =
            (A0 > 0.0)
            ?
            -std::log(U(rng))/A0
            :
            INF;

        const double t_rep1 =
            (ytot1 > 0)
            ?
            time_approx(
                t, ytot1, ytot,
                U(rng), p)
            :
            INF;

        const double t_rep2 =
            (ytot2 > 0)
            ?
            time_approx(
                t, ytot2, ytot,
                U(rng), p)
            :
            INF;

        double t_win;
        int wtype, wpole, wi, wj;

        if (t_rep1 <= t_rep2 &&
            t_rep1 <= t_hom)
        {
            t_win=t_rep1;
            wtype=0;
            wpole=0;
            wi=0;
            wj=0;
        }
        else if (t_rep2 <= t_hom)
        {
            t_win=t_rep2;
            wtype=0;
            wpole=1;
            wi=0;
            wj=0;
        }
        else if (A0 > 0.0)
        {
            t_win=t_hom;

            const double target =
                U(rng) * A0;

            double cum = 0.0;

            int idx =
                (int)rxns.size()-1;

            for (int r=0;
                 r<(int)rxns.size();
                 ++r)
            {
                cum += rxns[r].a;

                if (cum >= target) {
                    idx = r;
                    break;
                }
            }

            wtype = rxns[idx].type;
            wpole = rxns[idx].pole;
            wi    = rxns[idx].i;
            wj    = rxns[idx].j;
        }
        else break;

        Pole& yw =
            (wpole==0)? y1 : y2;

        std::vector<int>& actw =
            (wpole==0)? act1 : act2;

        switch(wtype) {

        case 0:
            yw[0] += 1;

            if (wpole==0)
                ytot1++;
            else
                ytot2++;

            ytot++;
            break;

        case 1:
            yw[0] -= 2;
            upd(yw,actw,2,+1);
            break;

        case 2:
            yw[0] -= 1;
            upd(yw,actw,wi,-1);
            upd(yw,actw,wi+1,+1);
            break;

        case 3:

            if (wi==wj) {
                upd(yw,actw,wi,-2);
                upd(yw,actw,wi+wj,+1);
            }
            else {
                upd(yw,actw,wi,-1);
                upd(yw,actw,wj,-1);
                upd(yw,actw,wi+wj,+1);
            }

            break;

        case 4:

            if (wi==2) {
                yw[0] += 2;
                upd(yw,actw,2,-1);
            }
            else {
                yw[0] += 1;
                upd(yw,actw,wi,-1);
                upd(yw,actw,wi-1,+1);
            }

            break;

        case 5:
            y1[0]--;
            y2[0]++;
            ytot1--;
            ytot2++;
            break;

        case 6:
            y2[0]--;
            y1[0]++;
            ytot2--;
            ytot1++;
            break;
        }

        t += t_win;
    }
}

// ------------------------------------------------------------

static long long lineage_to_loss(
    const Params& p,
    std::mt19937_64& rng)
{
    Pole y1(p.max_size,0),
         y2(p.max_size,0);

    y1[0] = p.x0;

    long long gen = 0;

    while (true) {

        if (pole_total(y1) == 0)
            break;

        ssa_cycle(y1,y2,p,rng);

        ++gen;

        if (rng() & 1)
            std::swap(y1,y2);

        std::fill(
            y2.begin(),
            y2.end(),
            0);
    }

    return gen;
}

// ------------------------------------------------------------

int main(int argc, char* argv[])
{
    if (argc < 5) {
        std::cerr
            << "usage:\n"
            << "./fig3b_cluster_only "
            << "tau n_lineages seed output.txt\n";
        return 1;
    }

Params p;

p.tau = std::atof(argv[1]);
p.mu  = std::log(2.0) / p.tau;

// generation-time-specific parameters
if ((int)p.tau == 24) {

    p.k_tr = 1.49;
    p.K    = 15.2 / 1.26;
    p.V0   = 1.6;

}
else if ((int)p.tau == 30) {

    p.k_tr = 2.17;
    p.K    = 15.8 / 1.26;
    p.V0   = 1.2;

}
else if ((int)p.tau == 40) {

    p.k_tr = 2.84;
    p.K    = 14.5 / 1.26;
    p.V0   = 0.8;

}
else {

    std::cerr << "Unsupported tau\n";
    return 1;

}
    const int n_lineages =
        std::atoi(argv[2]);

    const uint64_t seed =
        std::strtoull(argv[3],nullptr,10);

    const std::string outfile =
        argv[4];

    std::mt19937_64 rng(seed);

    std::ofstream fout(outfile);

    for (int i=0; i<n_lineages; ++i) {

        const long long g =
            lineage_to_loss(p,rng);

        fout << (int)p.tau
             << " "
             << g
             << "\n";
    }

    fout.close();

    return 0;
}
