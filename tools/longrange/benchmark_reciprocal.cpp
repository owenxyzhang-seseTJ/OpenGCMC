// Full versus conjugate-half storage, identical finite Ewald Hamiltonian.
#include "ewald_state.h"
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
using Clock=std::chrono::steady_clock;
int main(){
 constexpr std::size_t H=4416,A=12,B=10,repetitions=50;
 std::mt19937_64 rng(621112);std::uniform_real_distribution<double> random(0,40);
 std::vector<double>host(3*H),hq(H),trial(3*A*B),q(A);
 for(auto&v:host)v=random(rng);for(std::size_t i=0;i<H;++i)hq[i]=i%2?.08:-.08;
 for(auto&v:trial)v=random(rng);for(std::size_t i=0;i<A;++i)q[i]=i%2?.115:-.115;
 poremc::ReciprocalConfig cfg;cfg.cell={40,0,0,0,40,0,0,0,40};cfg.kmax={13,13,13};cfg.alpha=.28;
 std::cout<<std::setprecision(17)<<"{\"host_atoms\":"<<H<<",\"candidate_atoms\":"<<A<<",\"candidates\":"<<B<<",\"kmax\":[13,13,13],\"threads\":1,\"repetitions\":"<<repetitions<<",\"records\":[";
 bool first=true;double sink=0;
 for(int repeat=0;repeat<3;++repeat)for(int order=0;order<2;++order){
  bool half=bool((repeat+order)%2);cfg.half_spectrum=half;
  auto t0=Clock::now();poremc::EwaldState state(cfg,H,host.data(),hq.data());auto t1=Clock::now();
  for(std::size_t i=0;i<10;++i){auto a=state.contribution(A,trial.data()+3*A*i,q.data());state.commit(&a);}
  double output[B];for(int i=0;i<5;++i)state.batch_insertion(B,A,trial.data(),q.data(),nullptr,nullptr,output);
  auto t2=Clock::now();for(std::size_t i=0;i<repetitions;++i){state.batch_insertion(B,A,trial.data(),q.data(),nullptr,nullptr,output);sink+=output[i%B];}auto t3=Clock::now();
  auto field=state.host_potential_grid({64,64,64});auto t4=Clock::now();sink+=field[42];
  const std::array<double,3> displacement={.3,-.2,.4};
  auto original=state.contribution(A,trial.data(),q.data());std::vector<double> shifted(trial.begin(),trial.begin()+3*A);
  for(std::size_t a=0;a<A;++a)for(int d=0;d<3;++d)shifted[3*a+d]+=displacement[d];
  auto t5=Clock::now();for(std::size_t i=0;i<200;++i){auto a=state.contribution(A,shifted.data(),q.data());sink+=a.modes[0].real();}auto t6=Clock::now();
  for(std::size_t i=0;i<200;++i){auto a=state.translated_contribution(original,displacement);sink+=a.modes[0].real();}auto t7=Clock::now();
  if(!first)std::cout<<',';first=false;
  std::cout<<"{\"repeat\":"<<repeat<<",\"half_spectrum\":"<<(half?"true":"false")<<",\"stored_modes\":"<<state.stats().modes<<",\"host_build_s\":"<<std::chrono::duration<double>(t1-t0).count()<<",\"candidate_batch_s\":"<<std::chrono::duration<double>(t3-t2).count()/repetitions<<",\"host_fft64_s\":"<<std::chrono::duration<double>(t4-t3).count()<<",\"translation_full_recompute_s\":"<<std::chrono::duration<double>(t6-t5).count()/200<<",\"translation_phase_s\":"<<std::chrono::duration<double>(t7-t6).count()/200<<'}';
 }
 std::cout<<"],\"checksum\":"<<sink<<"}\n";
}
