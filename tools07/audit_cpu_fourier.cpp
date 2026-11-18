// Independent binary64 full-wave oracle for 0.7's direct single-site contraction
// and one-pass molecule replacement. Coordinates for the oracle are fractional.
#include "ewald_state.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
namespace {
using Vec=std::array<double,3>;
using C=std::complex<double>;
constexpr double pi=3.14159265358979323846, ck=167100.94689828736;
struct Atom {Vec f; double q;};
using Atoms=std::vector<Atom>;
struct Wave {std::array<int,3> n;double weight;};
Vec cross(Vec a,Vec b) {
 return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};
}
double dot(Vec a,Vec b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
struct Oracle {
 std::vector<Wave> waves;
 double self, bg;
 Oracle(const poremc::ReciprocalConfig& cfg) {
  Vec a,b,c;
  for(int i=0;i<3;++i){a[i]=cfg.cell[i];b[i]=cfg.cell[3+i];c[i]=cfg.cell[6+i];}
  const double volume=dot(a,cross(b,c));
  const std::array<Vec,3> dual={cross(b,c),cross(c,a),cross(a,b)};
  for(int i=-cfg.kmax[0];i<=cfg.kmax[0];++i)
   for(int j=-cfg.kmax[1];j<=cfg.kmax[1];++j)
    for(int k=-cfg.kmax[2];k<=cfg.kmax[2];++k) {
     if(i==0 && j==0 && k==0)continue;
     Vec wave;
     for(int d=0;d<3;++d)wave[d]=(2*pi/volume)*(i*dual[0][d]+j*dual[1][d]+k*dual[2][d]);
     const double k2=dot(wave,wave);
     if(cfg.k_cutoff && k2>cfg.k_cutoff*cfg.k_cutoff)continue;
     waves.push_back({{i,j,k},ck*2*pi/volume*std::exp(-k2/(4*cfg.alpha*cfg.alpha))/k2});
    }
  self=-ck*cfg.alpha/std::sqrt(pi);
  bg=cfg.background?-ck*pi/(2*cfg.alpha*cfg.alpha*volume):0.0;
 }
 double energy(const Atoms& atoms) const {
  long double sum=0;double q=0,q2=0;
  for(auto a:atoms){q+=a.q;q2+=a.q*a.q;}
  for(const auto& w:waves) {
   C s{};
   for(auto a:atoms) {
    double arg=2*pi*(w.n[0]*a.f[0]+w.n[1]*a.f[1]+w.n[2]*a.f[2]);
    s+=a.q*C(std::cos(arg),std::sin(arg));
   }
   sum+=w.weight*std::norm(s);
  }
  return static_cast<double>(sum)+self*q2+bg*q*q;
 }
};

struct NativeAtoms {
 std::vector<double> xyz,q;
 NativeAtoms(const Atoms& atoms, const std::array<double,9>& cell) {
  for(auto a:atoms) {
   for(int d=0;d<3;++d)xyz.push_back(a.f[0]*cell[d]+a.f[1]*cell[3+d]+a.f[2]*cell[6+d]);
   q.push_back(a.q);
  }
 }
 poremc::ChargeContribution contribution(const poremc::EwaldState& st) const {
  return st.contribution(q.size(),xyz.data(),q.data());
 }
};
Atoms combine(Atoms a,const Atoms& b){a.insert(a.end(),b.begin(),b.end());return a;}
void check(double value,double ref,double& maximum) {
 const double error=std::abs(value-ref);
 if(!std::isfinite(error)||error>1.e-7+5.e-12*std::abs(ref))throw std::runtime_error("Independent full-mode mismatch");
 maximum=std::max(error,maximum);
}
}
int main() {
 std::mt19937_64 rng(30718);
 std::uniform_real_distribution<double> position(-2,3),charge(-.25,.25);
 auto random_atoms=[&](int n){Atoms result;for(int i=0;i<n;++i)
  result.push_back({{position(rng),position(rng),position(rng)}, i%5==1?0.0:charge(rng)});return result;};
 double largest_single=0,largest_replacement=0,largest_empty=0;int configurations=0,batch_trials=0;
 for(bool half:{false,true})for(bool radial:{false,true})for(bool background:{false,true}) {
  poremc::ReciprocalConfig cfg;cfg.cell={36,0,0,-3,39,0,5,2,41};
  cfg.alpha=.26;cfg.background=background;cfg.half_spectrum=half;
  cfg.kmax=radial?std::array<int,3>{0,6,8}:std::array<int,3>{5,3,6};
  cfg.k_cutoff=radial?.72:0;cfg.threads=2;
  Oracle oracle(cfg);
  const auto host=random_atoms(7), old=random_atoms(17), next=random_atoms(19), prefix=random_atoms(6);
  auto bath=combine(host,old);
  NativeAtoms h(host,cfg.cell),o(old,cfg.cell),n(next,cfg.cell),p(prefix,cfg.cell);
  poremc::EwaldState state(cfg,h.q.size(),h.xyz.data(),h.q.data());
  auto former=o.contribution(state), after=n.contribution(state), start=p.contribution(state);
  state.commit(&former);
  auto new_bath=combine(host,next);
  check(state.replacement_energy(after,former), oracle.energy(new_bath)-oracle.energy(bath),largest_replacement);
  for(bool exclude:{false,true})for(bool grow:{false,true}) {
   auto environment=exclude?host:bath;
   if(grow)environment=combine(std::move(environment),prefix);
   const double base=oracle.energy(environment);
   for(int b:{1,3,4,11}) {
    auto candidates=random_atoms(b);
    const double q=radial?-.13:.24;
    for(auto& a:candidates)a.q=q;
    NativeAtoms trials(candidates,cfg.cell);
    std::vector<double> energy(b,-7.0);
    state.batch_insertion(b,1,trials.xyz.data(),&q,exclude?&former:nullptr,grow?&start:nullptr,energy.data());
    for(int i=0;i<b;++i) {
     const double ref=oracle.energy(combine(environment,Atoms{candidates[i]}))-base;
     check(energy[i],ref,largest_single);
    }
    const double zero=0;
    state.batch_insertion(b,1,trials.xyz.data(),&zero,exclude?&former:nullptr,grow?&start:nullptr,energy.data());
    for(auto value:energy)check(value,0,largest_empty);
    batch_trials+=b;
   }
  }
  ++configurations;
 }
 std::cout<<std::setprecision(16)<<"{\"configurations\":"<<configurations
 <<",\"single_site_candidates\":"<<batch_trials<<",\"max_single_site_error_K\":"<<largest_single
 <<",\"max_replacement_error_K\":"<<largest_replacement
 <<",\"max_uncharged_error_K\":"<<largest_empty<<"}\n";
}
