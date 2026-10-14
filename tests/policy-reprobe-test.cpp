#include "strata/spec/draft_policy.hpp"
#include <cstdlib>
#include <cstdio>

int main() {
    _putenv_s("STRATA_LOOKUP_REPROBE_ROUNDS", "");
    strata::spec::DraftPolicy baseline(4);
    _putenv_s("STRATA_LOOKUP_REPROBE_ROUNDS", "64");
    strata::spec::DraftPolicy candidate(4);
    auto teach = [](strata::spec::DraftPolicy& p) {
        for (int n=0; n<5; ++n) {
            p.observe(true, 3, 2, 32, 300); // cold SSD cost, before RAM placement warms
            p.observe(true, 4, 3, 32, 100);
        }
        for (int n=0; n<128; ++n) p.observe(true, 2, 1, 32, 10);
    };
    teach(baseline); teach(candidate);
    if (baseline.choose(1,3,32).t != 2) return 1;
    if (candidate.choose(1,3,32).t != 3) return 2;
    candidate.observe(true,3,2,32,12); // stale price must not retain the cold 300ms cost
    if (candidate.cost_ms(3) != 12) return 3;
    candidate.observe(true,4,3,32,100);
    if (candidate.choose(1,3,32).t != 3) return 4;
    if (baseline.choose(1,3,32).t != 2) return 5;
    _putenv_s("STRATA_LOOKUP_REPROBE_ROUNDS", "invalid");
    strata::spec::DraftPolicy invalid(4); teach(invalid);
    if (invalid.choose(1,3,32).t != 2) return 6;
    _putenv_s("STRATA_LOOKUP_REPROBE_ROUNDS", "64");
    strata::spec::DraftPolicy unprofitable_best(6);
    auto cold_all = [](strata::spec::DraftPolicy& p, bool moderate) {
        for(int j=0;j<5;++j) {
            p.observe(false,1,0,32,10);
            for(int t=2;t<=6;++t) p.observe(true,t,t-1,32,40.0*(t-1));
        }
        for(int j=0;j<128;++j) p.observe(moderate,moderate?5:1,moderate?3:0,32,moderate?160:10);
    };
    cold_all(unprofitable_best,false);
    if(!unprofitable_best.choose(1,5,32).lookup || unprofitable_best.choose(1,5,32).t!=2) return 9;
    std::puts("policy reprobe: cold-middle recovery and unchanged defaults PASS");
    return 0;
}
