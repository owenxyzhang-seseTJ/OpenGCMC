// Build unchanged against 0.5 (--define POREMC_CPU_VERSION=5), 0.6 or 0.7.
#include "ewald_state.h"
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#ifndef POREMC_CPU_VERSION
#define POREMC_CPU_VERSION 7
#endif
using Clock=std::chrono::steady_clock;
template<class F> double microseconds(int loops, F&& call, double& checksum) {
 for(int warm=0;warm<4;++warm) checksum+=call(warm);
 auto begin=Clock::now();
 for(int i=0;i<loops;++i) checksum+=call(i);
 return std::chrono::duration<double,std::micro>(Clock::now()-begin).count()/loops;
}
int main() {
 constexpr int H=4416,A=12,B=10,N=5;
 poremc::ReciprocalConfig cfg;
 cfg.cell={48,0,0,0,48,0,0,0,48}; cfg.alpha=.28; cfg.kmax={13,13,13};
 std::mt19937_64 rng(140509);
 std::uniform_real_distribution<double> uniformly(0,48);
 std::vector<double> hx(3*H), hq(H), xyz(3*A*B), q(A);
 for(double& v:hx) v=uniformly(rng);
 for(int a=0;a<H;++a) hq[a]=(a%2 ? -.08:.08);
 for(double& v:xyz) v=uniformly(rng);
 for(int a=0;a<A;++a) q[a]=a<8 ? (a%2 ? -.10:.10) : 0.0;
 const auto start=Clock::now();
 poremc::EwaldState state(cfg,H,hx.data(),hq.data());
 const double cold=std::chrono::duration<double,std::micro>(Clock::now()-start).count();
 poremc::ChargeContribution old;
 for(int i=0;i<N;++i) {auto molecule=state.contribution(A,xyz.data()+3*A*i,q.data());state.commit(&molecule);old=std::move(molecule);}
 auto fresh=state.contribution(A,xyz.data(),q.data());
 auto prefix=state.contribution(3,xyz.data(),q.data()); double keep=0;
 std::array<double,B> energies;
 const double molecular=microseconds(70,[&](int i){state.batch_insertion(B,A,xyz.data(),q.data(),&old,nullptr,energies.data());return energies[i%B];},keep);
 const double sites=microseconds(120,[&](int i){state.batch_insertion(B,1,xyz.data(),q.data(),&old,&prefix,energies.data());return energies[i%B];},keep);
 const double contribution=microseconds(250,[&](int i){auto r=state.contribution(A,xyz.data()+3*A*(i%B),q.data());return r.modes[i%17].real();},keep);
 const std::array<double,3> translation={.37,-.26,.52};
 std::vector<double> shifted(xyz.begin(),xyz.begin()+3*A);
 for(int a=0;a<A;++a)for(int d=0;d<3;++d)shifted[3*a+d]+=translation[d];
 const double translate=microseconds(500,[&](int i){
#if POREMC_CPU_VERSION >= 6
  auto result=state.translated_contribution(fresh,translation);
#else
  auto result=state.contribution(A,shifted.data(),q.data());
#endif
  return result.modes[i%17].real();
 },keep);
 const double replacement=microseconds(800,[&](int) {
#if POREMC_CPU_VERSION >= 7
  return state.replacement_energy(fresh,old);
#else
  return state.insertion_energy(fresh,&old)-state.insertion_energy(old,&old);
#endif
 },keep);
 const double deletion=microseconds(800,[&](int){return state.insertion_energy(old,&old);},keep);
 std::cout<<std::setprecision(17)<<"{\"version\":"<<POREMC_CPU_VERSION
 <<",\"modes\":"<<state.stats().modes<<",\"host_build_us\":"<<cold
 <<",\"rigid_10_candidates_us\":"<<molecular<<",\"single_site_10_trials_with_prefix_us\":"<<sites
 <<",\"selected_molecule_structure_us\":"<<contribution<<",\"replacement_energy_us\":"<<replacement
 <<",\"translated_structure_us\":"<<translate<<",\"removal_energy_us\":"<<deletion<<",\"checksum\":"<<keep<<"}\n";
}
