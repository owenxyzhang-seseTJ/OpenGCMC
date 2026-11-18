// Independent full-mode reference for the 0.6 half-spectrum and host FFT.
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
using C=std::complex<double>;
constexpr double pi=3.1415926535897932384626433832795;
constexpr double ck=167100.94689828736;
struct Reference {
 poremc::ReciprocalConfig cfg;
 std::vector<std::array<double,3>> waves;
 std::vector<double> weights;
 std::vector<C> host;
 double volume,qhost=0;
 Reference(const poremc::ReciprocalConfig& c,const std::vector<double>& x,const std::vector<double>& q):cfg(c){
  const auto& h=c.cell;
  volume=h[0]*(h[4]*h[8]-h[5]*h[7])-h[1]*(h[3]*h[8]-h[5]*h[6])+h[2]*(h[3]*h[7]-h[4]*h[6]);
  std::array<double,9> inv={(h[4]*h[8]-h[5]*h[7])/volume,(h[2]*h[7]-h[1]*h[8])/volume,(h[1]*h[5]-h[2]*h[4])/volume,
   (h[5]*h[6]-h[3]*h[8])/volume,(h[0]*h[8]-h[2]*h[6])/volume,(h[2]*h[3]-h[0]*h[5])/volume,
   (h[3]*h[7]-h[4]*h[6])/volume,(h[1]*h[6]-h[0]*h[7])/volume,(h[0]*h[4]-h[1]*h[3])/volume};
  for(int i=-c.kmax[0];i<=c.kmax[0];++i)for(int j=-c.kmax[1];j<=c.kmax[1];++j)for(int l=-c.kmax[2];l<=c.kmax[2];++l){
   if(i==0&&j==0&&l==0)continue;
   std::array<double,3> k={2*pi*(inv[0]*i+inv[1]*j+inv[2]*l),2*pi*(inv[3]*i+inv[4]*j+inv[5]*l),2*pi*(inv[6]*i+inv[7]*j+inv[8]*l)};
   double k2=k[0]*k[0]+k[1]*k[1]+k[2]*k[2];if(c.k_cutoff>0&&k2>c.k_cutoff*c.k_cutoff)continue;
   waves.push_back(k);weights.push_back(ck*2*pi/volume*std::exp(-k2/(4*c.alpha*c.alpha))/k2);
  }
  host=structure(x,q);for(double a:q)qhost+=a;
 }
 std::vector<C> structure(const std::vector<double>& x,const std::vector<double>& q)const{
  std::vector<C> s(waves.size(),C(0,0));
  for(std::size_t k=0;k<waves.size();++k)for(std::size_t a=0;a<q.size();++a)s[k]+=q[a]*std::exp(C(0,waves[k][0]*x[3*a]+waves[k][1]*x[3*a+1]+waves[k][2]*x[3*a+2]));
  return s;
 }
 double energy(const std::vector<double>&x,const std::vector<double>&q)const{
  auto s=structure(x,q);double e=0,Q=0,Q2=0;
  for(std::size_t k=0;k<s.size();++k)e+=weights[k]*(2*std::real(std::conj(host[k])*s[k])+std::norm(s[k]));
  for(double a:q){Q+=a;Q2+=a*a;}
  e-=ck*cfg.alpha/std::sqrt(pi)*Q2;
  if(cfg.background)e-=ck*pi/(2*cfg.alpha*cfg.alpha*volume)*(2*qhost*Q+Q*Q);
  return e;
 }
 double field(const double*x)const{
  double e=0;for(std::size_t k=0;k<waves.size();++k)e+=2*weights[k]*std::real(std::conj(host[k])*std::exp(C(0,waves[k][0]*x[0]+waves[k][1]*x[1]+waves[k][2]*x[2])));return e;
 }
};
int main(){try{
 std::mt19937_64 rng(622817);std::uniform_real_distribution<double> uniform(0,1);
 double max_energy=0,max_grid=0,max_field=0,max_delta=0,max_pme_grid=0,max_translation=0;
 std::size_t configurations=0,nodes=0;
 auto check=[](double a,double b,double& error){double d=std::abs(a-b);error=std::max(error,d);if(d>2e-8+2e-12*std::max(std::abs(a),std::abs(b)))throw std::runtime_error("independent reciprocal comparison failed");};
 for(int triclinic=0;triclinic<2;++triclinic)for(double cutoff:{0.,.81}){
  poremc::ReciprocalConfig cfg;cfg.cell={31,0,0,triclinic?3.:0.,33,0,triclinic?-2.:0.,triclinic?4.:0.,35};cfg.alpha=.3;cfg.kmax={5,4,3};cfg.k_cutoff=cutoff;
  auto positions=[&](int n){std::vector<double>x(3*n);for(int a=0;a<n;++a){double f[3]={3*uniform(rng)-1,3*uniform(rng)-1,3*uniform(rng)-1};for(int d=0;d<3;++d)x[3*a+d]=f[0]*cfg.cell[d]+f[1]*cfg.cell[3+d]+f[2]*cfg.cell[6+d];}return x;};
  auto hx=positions(9),x=positions(3),y=positions(3);std::vector<double>hq={.13,-.17,.11,-.09,.14,-.16,.03,-.08,.04},q={.3,-.17,.07};
  Reference ref(cfg,hx,hq);
  for(bool half:{false,true}){
   cfg.half_spectrum=half;poremc::EwaldState state(cfg,hq.size(),hx.data(),hq.data());
   if(state.stats().full_modes!=ref.waves.size()||state.stats().modes*(half?2:1)!=ref.waves.size())throw std::runtime_error("mode statistics mismatch");
   auto a=state.contribution(q.size(),x.data(),q.data()),b=state.contribution(q.size(),y.data(),q.data());
   const std::array<double,3> displacement={40.7,-76.3,5.9};auto shifted=x;
   for(std::size_t i=0;i<q.size();++i)for(int d=0;d<3;++d)shifted[3*i+d]+=displacement[d];
   auto translated=state.translated_contribution(a,displacement),recomputed=state.contribution(q.size(),shifted.data(),q.data());
   for(std::size_t k=0;k<translated.modes.size();++k)check(std::abs(translated.modes[k]-recomputed.modes[k]),0.,max_translation);
   check(state.insertion_energy(translated),ref.energy(shifted,q),max_delta);
   check(state.insertion_energy(a),ref.energy(x,q),max_energy);
   double host_energy=0;for(std::size_t i=0;i<q.size();++i)host_energy+=q[i]*ref.field(x.data()+3*i);
   check(state.host_insertion_energy(a),host_energy,max_field);
   check(state.host_insertion_energy(q.size(),x.data(),q.data()),host_energy,max_field);
   state.commit(&a);check(state.total_host_relative_energy(),ref.energy(x,q),max_energy);
   check(state.insertion_energy(b,&a),ref.energy(y,q),max_delta);
   state.commit(&b,&a);check(state.total_host_relative_energy(),ref.energy(y,q),max_energy);
   auto saved=state.serialize();poremc::EwaldState restored(cfg,hq.size(),hx.data(),hq.data());restored.restore(saved);
   if(restored.total_host_relative_energy()!=state.total_host_relative_energy())throw std::runtime_error("checkpoint mismatch");
   for(std::array<int,3> mesh:{std::array<int,3>{8,8,8},std::array<int,3>{16,32,16}}){
    auto field=state.host_potential_grid(mesh);
    for(int i=0;i<mesh[0];++i)for(int j=0;j<mesh[1];j+=3)for(int k=0;k<mesh[2];k+=3){
     double f[3]={double(i)/mesh[0],double(j)/mesh[1],double(k)/mesh[2]},r[3];for(int d=0;d<3;++d)r[d]=f[0]*cfg.cell[d]+f[1]*cfg.cell[3+d]+f[2]*cfg.cell[6+d];
     check(field[(std::size_t(i)*mesh[1]+j)*mesh[2]+k],ref.field(r),max_grid);++nodes;
    }
   }
   ++configurations;
  }
  cfg.method=1;cfg.k_cutoff=0;cfg.mesh={16,16,16};poremc::EwaldState pme(cfg,hq.size(),hx.data(),hq.data());
  auto pa=pme.contribution(q.size(),x.data(),q.data());double before=pme.host_insertion_energy(q.size(),x.data(),q.data());pme.commit(&pa);
  if(before!=pme.host_insertion_energy(q.size(),x.data(),q.data()))throw std::runtime_error("PME host cache changed on guest commit");
  auto field=pme.host_potential_grid({8,16,8});const double one=1.;
  for(int i=0;i<8;++i)for(int j=0;j<16;j+=3)for(int k=0;k<8;k+=3){
   double f[3]={double(i)/8,double(j)/16,double(k)/8},r[3];for(int d=0;d<3;++d)r[d]=f[0]*cfg.cell[d]+f[1]*cfg.cell[3+d]+f[2]*cfg.cell[6+d];
   auto a=pme.contribution(1,r,&one);check(field[(std::size_t(i)*16+j)*8+k],pme.host_insertion_energy(a),max_pme_grid);
  }
 }
 // Report the task's complete rectangular mode count and its exact pairing.
 poremc::ReciprocalConfig large;large.cell={40,0,0,0,40,0,0,0,40};large.kmax={13,13,13};poremc::EwaldState paired(large,0,nullptr,nullptr);
 if(paired.stats().full_modes!=19682||paired.stats().modes!=9841)throw std::runtime_error("kmax=13 mode count mismatch");
 std::cout<<std::setprecision(17)<<"{\"direct_configurations\":"<<configurations<<",\"host_fft_nodes\":"<<nodes<<",\"max_total_energy_error_K\":"<<max_energy<<",\"max_increment_error_K\":"<<max_delta<<",\"max_translation_structure_factor_error\":"<<max_translation<<",\"max_host_field_error_K\":"<<max_field<<",\"max_host_fft_error_K\":"<<max_grid<<",\"max_pme_host_grid_error_K\":"<<max_pme_grid<<",\"kmax13_full_modes\":"<<paired.stats().full_modes<<",\"kmax13_stored_modes\":"<<paired.stats().modes<<"}\n";
 return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
