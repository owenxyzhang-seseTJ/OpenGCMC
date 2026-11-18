// Independently derived, full C++ Monte Carlo loop. No upstream code copied.
#include "fast_mc.h"
#include "ewald_state.h"
#include "host_grid.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <iomanip>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
extern "C" {
void* poremc_environment_create(size_t,const double*,const double*,const double*,const double*,const double*,double,int,double,int,int,int);
void poremc_environment_destroy(void*);
int poremc_environment_energy(void*,size_t,size_t,const double*,const double*,const double*,const double*,double*);
int poremc_environment_energy_excluding(void*,size_t,size_t,const double*,const double*,const double*,const double*,size_t,size_t,double*);
const char* poremc_last_error();
void* poremc_query_create(void*,size_t,const double*,const double*,const double*);
void poremc_query_destroy(void*);
int poremc_query_energy(void*,size_t,const double*,double*);
}
namespace {
constexpr double PI=3.1415926535897932384626433832795;
constexpr double CK=167100.94689828736;
thread_local std::string fast_error;
struct V {double x,y,z;};
V add(V a,V b){return {a.x+b.x,a.y+b.y,a.z+b.z};}
V sub(V a,V b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
V mul(V a,double s){return {a.x*s,a.y*s,a.z*s};}
double dot(V a,V b){return a.x*b.x+a.y*b.y+a.z*b.z;}
V cross(V a,V b){return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
double norm(V a){return std::sqrt(dot(a,a));}
V unit(V a){double n=norm(a);if(n<1e-14)throw std::runtime_error("degenerate molecular reference axis");return mul(a,1/n);}
V get(const std::vector<double>& x,size_t i){return {x[3*i],x[3*i+1],x[3*i+2]};}
void put(std::vector<double>& x,size_t i,V v){x[3*i]=v.x;x[3*i+1]=v.y;x[3*i+2]=v.z;}
V row(V a,const std::array<double,9>& h){return {a.x*h[0]+a.y*h[3]+a.z*h[6],a.x*h[1]+a.y*h[4]+a.z*h[7],a.x*h[2]+a.y*h[5]+a.z*h[8]};}
using Env=std::unique_ptr<void,void(*)(void*)>;
struct Growth {std::vector<double> x;double logw=0,energy=0;bool valid=true;};
struct Fast {
 size_t A,ntors,max_particles;
 double T,z,rc,alpha,max_translation,max_rotation,scale14lj,scale14q,Vbox;
 int shifted,coulomb,algorithm,threads;bool rigid;
 std::array<double,9> H,inv;std::array<double,5> probability;
 std::vector<double> sigma,epsilon,charge,bond,angle,body,brange,arange,bk,ak,tk,tp;
 std::vector<int32_t> parent,aref,dref,K,excluded,tidx,tn,graph;
 std::vector<std::vector<double>> particles;
 // The immutable host is transformed once. Each accepted molecule has its
 // own contribution, so excluding an old molecule never transforms a bath.
 std::unique_ptr<poremc::EwaldState> longrange;
 std::unique_ptr<poremc::HostGrid> host_grid;
 bool delayed=false;
 double root_bias=0;
 // For each move: stage-1 evaluated, stage-1 passed, exact evaluations,
 // stage-2 accepted. Diagnostic only; never influences sampling decisions.
 std::array<uint64_t,20> acceleration_counters{};
 std::vector<poremc::ChargeContribution> particle_charge;
 std::vector<double> particle_host_internal;
 std::vector<double> particle_reference_local;
 std::vector<double> scratch_reciprocal;
 std::mt19937_64 rng;
 std::array<uint64_t,10> counters{};
 uint64_t steps=0;
 void* host;
 Env guest_cache{nullptr,poremc_environment_destroy};
 bool guest_dirty=true;
 size_t excluded_particle=std::numeric_limits<size_t>::max();
 std::uint64_t guest_builds=0,guest_reuses=0;
 std::vector<double> scratch_other,scratch_trial,scratch_energy,scratch_values,scratch_weights;
 Env host_query{nullptr,poremc_query_destroy};
 std::vector<Env> host_sites;
 std::vector<std::vector<size_t>> torsions_at;
 std::vector<double> bond_cube_lo,bond_cube_span,angle_cos,angle_cos_lo,angle_cos_span;
 std::vector<double> root_probability,root_threshold;
 std::vector<size_t> root_alias;
 std::array<int,3> root_shape{};
 double dsf_value=0,dsf_slope=0;
 Fast(const PoreMCFastConfig& c):A(c.atoms),ntors(c.torsions),max_particles(c.max_particles),T(c.temperature),z(c.activity),rc(c.cutoff),alpha(c.alpha),max_translation(c.max_translation),max_rotation(c.max_rotation),scale14lj(c.scale14_lj),scale14q(c.scale14_coulomb),shifted(c.shifted),coulomb(c.coulomb),algorithm(c.algorithm),threads(c.threads),rigid(c.rigid),rng(c.seed),host(c.host) {
  dsf_value=std::erfc(alpha*rc)/rc;dsf_slope=std::erfc(alpha*rc)/(rc*rc)+2*alpha/std::sqrt(PI)*std::exp(-alpha*alpha*rc*rc)/rc;
  if(A==0||max_particles==0||!host||!(T>0)||!std::isfinite(T)||z<0||!std::isfinite(z))throw std::invalid_argument("invalid FastGCMC thermodynamics/configuration");
  if(coulomb<0||coulomb>3)throw std::invalid_argument("unknown native Coulomb method");
  std::copy(c.cell,c.cell+9,H.begin());
  const double *h=H.data();Vbox=h[0]*(h[4]*h[8]-h[5]*h[7])-h[1]*(h[3]*h[8]-h[5]*h[6])+h[2]*(h[3]*h[7]-h[4]*h[6]);
  if(!(Vbox>0))throw std::invalid_argument("invalid cell");
  inv={(h[4]*h[8]-h[5]*h[7])/Vbox,(h[2]*h[7]-h[1]*h[8])/Vbox,(h[1]*h[5]-h[2]*h[4])/Vbox,(h[5]*h[6]-h[3]*h[8])/Vbox,(h[0]*h[8]-h[2]*h[6])/Vbox,(h[2]*h[3]-h[0]*h[5])/Vbox,(h[3]*h[7]-h[4]*h[6])/Vbox,(h[1]*h[6]-h[0]*h[7])/Vbox,(h[0]*h[4]-h[1]*h[3])/Vbox};
  auto dv=[&](const double*p,size_t n){return std::vector<double>(p,p+n);};auto iv=[&](const int32_t*p,size_t n){return std::vector<int32_t>(p,p+n);};
  sigma=dv(c.sigma,A);epsilon=dv(c.epsilon,A);charge=dv(c.charge,A);bond=dv(c.bond_lengths,A);angle=dv(c.angles,A);body=dv(c.rigid_positions,3*A);
  brange=dv(c.bond_ranges,2*A);arange=dv(c.angle_ranges,2*A);bk=dv(c.bond_constants,A);ak=dv(c.angle_constants,A);
  parent=iv(c.parents,A);aref=iv(c.angle_refs,A);dref=iv(c.dihedral_refs,A);K=iv(c.trials,A);excluded=iv(c.exclusions,A*A);
  tidx=iv(c.torsion_indices,4*ntors);tn=iv(c.torsion_periodicity,ntors);tk=dv(c.torsion_k,ntors);tp=dv(c.torsion_phase,ntors);
  double sum=0;for(size_t m=0;m<5;++m){probability[m]=c.move_weights[m];if(probability[m]<0||!std::isfinite(probability[m]))throw std::invalid_argument("invalid move probability");sum+=probability[m];}
  if(!(sum>0)||bool(probability[0])!=bool(probability[1]))throw std::invalid_argument("insertion and deletion weights must both be positive or both zero");
  for(auto&p:probability)p/=sum;
  for(auto k:K)if(k<1)throw std::invalid_argument("invalid CBMC trial count");
  graph.assign(A*A,static_cast<int32_t>(A+1));for(size_t i=0;i<A;++i){graph[i*A+i]=0;if(parent[i]>=0){graph[i*A+parent[i]]=1;graph[parent[i]*A+i]=1;}}
  for(size_t k=0;k<A;++k)for(size_t i=0;i<A;++i)for(size_t j=0;j<A;++j)graph[i*A+j]=std::min(graph[i*A+j],graph[i*A+k]+graph[k*A+j]);
  host_query=Env(poremc_query_create(host,A,sigma.data(),epsilon.data(),charge.data()),poremc_query_destroy);
  if(!host_query)throw std::runtime_error(poremc_last_error());
  for(size_t i=0;i<A;++i){host_sites.emplace_back(poremc_query_create(host,1,&sigma[i],&epsilon[i],&charge[i]),poremc_query_destroy);if(!host_sites.back())throw std::runtime_error(poremc_last_error());}
  torsions_at.resize(A);for(size_t t=0;t<ntors;++t)torsions_at[*std::max_element(tidx.begin()+4*t,tidx.begin()+4*t+4)].push_back(t);
  bond_cube_lo.resize(A);bond_cube_span.resize(A);angle_cos.resize(A);angle_cos_lo.resize(A);angle_cos_span.resize(A);
  for(size_t i=0;i<A;++i){double lo=brange[2*i],hi=brange[2*i+1];bond_cube_lo[i]=lo*lo*lo;bond_cube_span[i]=hi*hi*hi-lo*lo*lo;angle_cos[i]=std::cos(angle[i]);angle_cos_lo[i]=std::cos(arange[2*i+1]);angle_cos_span[i]=std::cos(arange[2*i])-angle_cos_lo[i];}
  if(coulomb>=2){
   poremc::ReciprocalConfig cfg;cfg.cell=H;cfg.alpha=alpha;cfg.threads=threads;
   cfg.half_spectrum=c.ewald_half_spectrum;cfg.k_cutoff=c.ewald_k_cutoff;
   for(int d=0;d<3;++d){cfg.kmax[d]=c.ewald_kmax[d];cfg.mesh[d]=c.pme_mesh[d];}
   cfg.method=c.reciprocal_method;cfg.backend=c.reciprocal_backend;
   if(cfg.method!=(coulomb==3?1:0))throw std::invalid_argument("Coulomb and reciprocal method disagree");
   if(c.cuda_library)cfg.cuda_library=c.cuda_library;
   longrange=std::make_unique<poremc::EwaldState>(cfg,c.host_atoms,c.host_positions,c.host_charge);
  }
  if(c.sampling_mode<0||c.sampling_mode>1)throw std::invalid_argument("unknown sampling mode");
  delayed=c.sampling_mode==1;root_bias=c.grid_root_bias;
  if(delayed){
   poremc::HostGridConfig gc;gc.cell=H;gc.threads=threads;gc.surrogate_cap=c.host_grid_cap;
   gc.temperature=T;gc.steep_interpolation=c.host_grid_steep;
   for(int d=0;d<3;++d)gc.mesh[d]=c.host_grid_mesh[d];
   if(c.host_grid_cache)gc.cache_path=c.host_grid_cache;
   if(c.host_grid_key)gc.cache_key=c.host_grid_key;
   poremc::HostReciprocalGridBuilder build;
   poremc::HostReciprocalEnergy exact;
   if(longrange){
    build=[this,gc](){return longrange->host_potential_grid(gc.mesh);};
    exact=[this](size_t n,const double*x,const double*q){return longrange->host_insertion_energy(n,x,q);};
   }
   host_grid=std::make_unique<poremc::HostGrid>(gc,host,A,sigma.data(),epsilon.data(),charge.data(),build,exact);
   if(root_bias>0)build_grid_root(root_bias);
  }
 }
 void build_grid_root(double bias){
  if(!(bias>0&&bias<1))throw std::invalid_argument("grid_root_bias must be in [0,1)");
  auto m=host_grid->config().mesh;size_t n=size_t(m[0])*m[1]*m[2];
  std::vector<double> xyz(3*n),u(n),p(n);size_t k=0;
  for(int i=0;i<m[0];++i)for(int j=0;j<m[1];++j)for(int l=0;l<m[2];++l){
   put(xyz,k++,row({(i+.5)/m[0],(j+.5)/m[1],(l+.5)/m[2]},H));
  }
  host_grid->batch_energy(n,1,xyz.data(),0,u.data());
  double minimum=*std::min_element(u.begin(),u.end()),sum=0;
  for(size_t i=0;i<n;++i){p[i]=std::exp(-(u[i]-minimum)/T);sum+=p[i];}
  if(!(sum>0)||!std::isfinite(sum))throw std::runtime_error("invalid host-grid root proposal");
  for(double&v:p)v=(1-bias)/n+bias*v/sum;
  set_root(m[0],m[1],m[2],p.data());
 }
 double uniform(){return std::generate_canonical<double,53>(rng);}
 void set_root(int nx,int ny,int nz,const double*p) {
  if(steps!=0)throw std::invalid_argument("root proposal must be fixed before production");
  if(nx<1||ny<1||nz<1)throw std::invalid_argument("invalid proposal grid");
  size_t n=size_t(nx)*ny*nz;root_shape={nx,ny,nz};root_probability.assign(p,p+n);
  double total=0;for(double v:root_probability){if(!(v>0)||!std::isfinite(v))throw std::invalid_argument("proposal must have full finite support");total+=v;}
  if(!std::isfinite(total))throw std::invalid_argument("proposal overflow");
  for(double&v:root_probability)v/=total;
  std::vector<double> scaled(n);std::vector<size_t> small,large;root_alias.resize(n);root_threshold.assign(n,1.);
  for(size_t i=0;i<n;++i){scaled[i]=root_probability[i]*n;root_alias[i]=i;(scaled[i]<1?small:large).push_back(i);}
  while(!small.empty()&&!large.empty()){
   size_t a=small.back(),b=large.back();small.pop_back();large.pop_back();root_threshold[a]=scaled[a];root_alias[a]=b;scaled[b]-=1-scaled[a];(scaled[b]<1?small:large).push_back(b);
  }
 }
 V root_position(){
  if(root_probability.empty())return row({uniform(),uniform(),uniform()},H);
  double u=uniform()*root_probability.size();size_t i=std::min(root_probability.size()-1,size_t(u));
  size_t k=u-i<root_threshold[i]?i:root_alias[i];
  int iz=k%root_shape[2],iy=(k/root_shape[2])%root_shape[1],ix=k/(size_t(root_shape[2])*root_shape[1]);
  return row({(ix+uniform())/root_shape[0],(iy+uniform())/root_shape[1],(iz+uniform())/root_shape[2]},H);
 }
 double root_logq(V x)const{
  if(root_probability.empty())return 0.;V f=row(x,inv);double v[3]={f.x,f.y,f.z};int key[3];
  for(int d=0;d<3;++d){v[d]-=std::floor(v[d]);key[d]=std::min(root_shape[d]-1,int(v[d]*root_shape[d]));}
  size_t k=(size_t(key[0])*root_shape[1]+key[1])*root_shape[2]+key[2];return std::log(root_probability.size()*root_probability[k]);
 }
 void* cached_environment(size_t exclude=std::numeric_limits<size_t>::max()){
  excluded_particle=exclude;
  if(guest_dirty || !guest_cache){guest_cache=environment();guest_dirty=false;++guest_builds;}
  else ++guest_reuses;
  return guest_cache.get();
 }

 double normal(){double u=std::max(uniform(),std::numeric_limits<double>::min());return std::sqrt(-2*std::log(u))*std::cos(2*PI*uniform());}
 V direction(){double c=2*uniform()-1,p=2*PI*uniform(),s=std::sqrt(std::max(0.,1-c*c));return {s*std::cos(p),s*std::sin(p),c};}
 std::array<double,9> rotation(){double w=normal(),x=normal(),y=normal(),zz=normal(),n=std::sqrt(w*w+x*x+y*y+zz*zz);w/=n;x/=n;y/=n;zz/=n;return {1-2*(y*y+zz*zz),2*(x*y-w*zz),2*(x*zz+w*y),2*(x*y+w*zz),1-2*(x*x+zz*zz),2*(y*zz-w*x),2*(x*zz-w*y),2*(y*zz+w*x),1-2*(x*x+y*y)};}
 V apply_rotation(V x,const std::array<double,9>&r){return {r[0]*x.x+r[1]*x.y+r[2]*x.z,r[3]*x.x+r[4]*x.y+r[5]*x.z,r[6]*x.x+r[7]*x.y+r[8]*x.z};}
 void wrap(std::vector<double>& x){V f=row(get(x,0),inv);V d=row({std::floor(f.x),std::floor(f.y),std::floor(f.z)},H);for(size_t i=0;i<A;++i)put(x,i,sub(get(x,i),d));}
 std::vector<double> propose_rigid(){V root=root_position();if(A==1)return {root.x,root.y,root.z};auto rot=rotation();V origin=root_probability.empty()?V{0,0,0}:apply_rotation(get(body,0),rot);std::vector<double>x(3*A);for(size_t i=0;i<A;++i)put(x,i,add(sub(apply_rotation(get(body,i),rot),origin),root));return x;}
 V propose(const std::vector<double>& prefix,size_t i){
  if(i==0)return root_position();
  double r=bond[i];if(brange[2*i+1]>brange[2*i]){double lo=brange[2*i],hi=brange[2*i+1];r=std::cbrt(lo*lo*lo+uniform()*(hi*hi*hi-lo*lo*lo));}
  V p=get(prefix,parent[i]);if(aref[i]<0)return add(p,mul(direction(),r));
  V e1=unit(sub(get(prefix,aref[i]),p));
  V v=dref[i]>=0?sub(get(prefix,dref[i]),get(prefix,aref[i])):(std::fabs(e1.x)<=std::fabs(e1.y)&&std::fabs(e1.x)<=std::fabs(e1.z)?V{1,0,0}:(std::fabs(e1.y)<=std::fabs(e1.z)?V{0,1,0}:V{0,0,1}));
  V e2=sub(v,mul(e1,dot(v,e1)));
  if(norm(e2)<1e-12){v=std::fabs(e1.x)<=std::fabs(e1.y)&&std::fabs(e1.x)<=std::fabs(e1.z)?V{1,0,0}:(std::fabs(e1.y)<=std::fabs(e1.z)?V{0,1,0}:V{0,0,1});e2=sub(v,mul(e1,dot(v,e1)));}
  e2=unit(e2);V e3=cross(e1,e2);double cosine=std::cos(angle[i]);
  if(arange[2*i+1]>arange[2*i])cosine=std::cos(arange[2*i+1])+uniform()*(std::cos(arange[2*i])-std::cos(arange[2*i+1]));
  double phi=2*PI*uniform()-PI,sine=std::sqrt(std::max(0.,1-cosine*cosine));
  return add(p,mul(add(mul(e1,cosine),mul(add(mul(e2,std::cos(phi)),mul(e3,std::sin(phi))),sine)),r));
 }
 void propose_batch(const std::vector<double>&prefix,size_t i,const std::vector<double>*old,std::vector<double>&trial){
  trial.resize(3*K[i]);size_t begin=0;if(old){put(trial,0,get(*old,i));begin=1;}
  V origin{0,0,0},e1{0,0,0},e2{0,0,0},e3{0,0,0};
  if(i){origin=get(prefix,parent[i]);if(aref[i]>=0){
   e1=unit(sub(get(prefix,aref[i]),origin));
   V v=dref[i]>=0?sub(get(prefix,dref[i]),get(prefix,aref[i])):(std::fabs(e1.x)<=std::fabs(e1.y)&&std::fabs(e1.x)<=std::fabs(e1.z)?V{1,0,0}:(std::fabs(e1.y)<=std::fabs(e1.z)?V{0,1,0}:V{0,0,1}));
   e2=sub(v,mul(e1,dot(v,e1)));if(norm(e2)<1e-12){v=std::fabs(e1.x)<=std::fabs(e1.y)&&std::fabs(e1.x)<=std::fabs(e1.z)?V{1,0,0}:(std::fabs(e1.y)<=std::fabs(e1.z)?V{0,1,0}:V{0,0,1});e2=sub(v,mul(e1,dot(v,e1)));}
   e2=unit(e2);e3=cross(e1,e2);
  }}
  for(size_t j=begin;j<size_t(K[i]);++j){
   if(i==0){put(trial,j,root_position());continue;}
   double r=bond[i];if(brange[2*i+1]>brange[2*i])r=std::cbrt(bond_cube_lo[i]+uniform()*bond_cube_span[i]);
   if(aref[i]<0){put(trial,j,add(origin,mul(direction(),r)));continue;}
   double cosine=angle_cos[i];if(arange[2*i+1]>arange[2*i])cosine=angle_cos_lo[i]+uniform()*angle_cos_span[i];
   double phi=2*PI*uniform()-PI,sine=std::sqrt(std::max(0.,1-cosine*cosine));
   put(trial,j,add(origin,mul(add(mul(e1,cosine),mul(add(mul(e2,std::cos(phi)),mul(e3,std::sin(phi))),sine)),r)));
  }
 }
 Env environment(size_t exclude=std::numeric_limits<size_t>::max(),size_t limit=std::numeric_limits<size_t>::max()){
  std::vector<double>p,s,e,q;size_t used=std::min(particles.size(),limit);p.reserve(used*A*3);s.reserve(used*A);e.reserve(used*A);q.reserve(used*A);
  for(size_t i=0;i<used;++i)if(i!=exclude){p.insert(p.end(),particles[i].begin(),particles[i].end());s.insert(s.end(),sigma.begin(),sigma.end());e.insert(e.end(),epsilon.begin(),epsilon.end());q.insert(q.end(),charge.begin(),charge.end());}
  void* handle=poremc_environment_create(s.size(),p.data(),s.data(),e.data(),q.data(),H.data(),rc,std::min(coulomb,2),alpha,shifted,algorithm==2?1:algorithm,threads);
  if(!handle)throw std::runtime_error(poremc_last_error());return Env(handle,poremc_environment_destroy);
 }
 void external(void* env,const double* positions,size_t batches,size_t atoms,const double*s,const double*e,const double*q,std::vector<double>&out,bool reference=false){
  out.resize(batches);scratch_other.resize(batches);auto&other=scratch_other;
  size_t exlo=0,exhi=0;if(env==guest_cache.get()&&excluded_particle<particles.size()){exlo=excluded_particle*A;exhi=exlo+A;}
  void* binding=atoms==A?host_query.get():host_sites[size_t(s-sigma.data())].get();
  if(reference&&host_grid)host_grid->batch_energy(batches,atoms,positions,size_t(s-sigma.data()),out.data());
  else if(poremc_query_energy(binding,batches,positions,out.data()))throw std::runtime_error(poremc_last_error());
  if(poremc_environment_energy_excluding(env,batches,atoms,positions,s,e,q,exlo,exhi,other.data()))throw std::runtime_error(poremc_last_error());
  for(size_t b=0;b<batches;++b)out[b]+=other[b];
 }
 double internal(const std::vector<double>&prefix,V x,size_t i,bool reference=false){
  if(rigid&&!longrange)return 0.;double u=0.;
  for(size_t j=0;j<i;++j){double r=norm(sub(x,get(prefix,j)));bool ex=rigid||excluded[i*A+j];double lj=ex?0.:(graph[i*A+j]==3?scale14lj:1.);double cq=ex?0.:(graph[i*A+j]==3?scale14q:1.);
   if(r<rc){double s=.5*(sigma[i]+sigma[j]),e=std::sqrt(epsilon[i]*epsilon[j]);
    if(lj&&s&&e){if(r<=1e-12)return std::numeric_limits<double>::infinity();double t=s/r,t6=std::pow(t,6),v=4*e*t6*(t6-1);if(shifted){double c6=std::pow(s/rc,6);v-=4*e*c6*(c6-1);}u+=lj*v;}
    double qq=charge[i]*charge[j];if(coulomb==1&&cq&&qq){if(r<=1e-12)return std::numeric_limits<double>::infinity();u+=cq*CK*qq*(std::erfc(alpha*r)/r-dsf_value+dsf_slope*(r-rc));}
   }
   if(longrange&&charge[i]*charge[j]!=0.){
    // Ewald includes every charge in reciprocal space. Remove the *bare*
    // primary interaction for exclusions / 1-4 scaling, retaining images.
    // Molecular coordinates are unwrapped, never minimum-imaged here.
    double radial=0.;
    if(r<=1e-12){if(cq!=0.)return std::numeric_limits<double>::infinity();radial=reference?0.:-2*alpha/std::sqrt(PI);}
    else if(r<rc)radial=reference?cq/r:(cq==0.?-std::erf(alpha*r):std::erfc(alpha*r)-(1-cq))/r;
    else radial=reference?(std::erf(alpha*r)-(1-cq))/r:-(1-cq)/r;
    u+=CK*charge[i]*charge[j]*radial;
   }
  }
  if(rigid)return u;
  if(i){double r=norm(sub(x,get(prefix,parent[i])));u+=.5*bk[i]*(r-bond[i])*(r-bond[i]);if(aref[i]>=0&&ak[i]){V a=sub(x,get(prefix,parent[i])),b=sub(get(prefix,aref[i]),get(prefix,parent[i]));double co=std::clamp(dot(a,b)/(norm(a)*norm(b)),-1.,1.);double delta=std::acos(co)-angle[i];u+=.5*ak[i]*delta*delta;}}
  for(size_t t:torsions_at[i]){V p[4];for(size_t k=0;k<4;++k)p[k]=tidx[4*t+k]==static_cast<int>(i)?x:get(prefix,tidx[4*t+k]);V b0=sub(p[0],p[1]),b1=unit(sub(p[2],p[1])),b2=sub(p[3],p[2]);V v=sub(b0,mul(b1,dot(b0,b1))),w=sub(b2,mul(b1,dot(b2,b1)));double phi=std::atan2(dot(cross(b1,v),w),dot(v,w));u+=tk[t]*(1+std::cos(tn[t]*phi-tp[t]));}
  return u;
 }
 const poremc::ChargeContribution* excluded_charge(void*env)const{
  return longrange&&env==guest_cache.get()&&excluded_particle<particle_charge.size()?&particle_charge[excluded_particle]:nullptr;
 }
 void reciprocal_batch(void*env,size_t batches,size_t atoms,const double*xyz,const double*q,
                       const poremc::ChargeContribution*prefix,std::vector<double>&energies){
  if(!longrange)return;
  scratch_reciprocal.resize(batches);
  longrange->batch_insertion(batches,atoms,xyz,q,excluded_charge(env),prefix,scratch_reciprocal.data());
  for(size_t b=0;b<batches;++b)energies[b]+=scratch_reciprocal[b];
 }
 void site_energies(void* env,const std::vector<double>&prefix,size_t i,const std::vector<double>&trial,
                    std::vector<double>&energies,const poremc::ChargeContribution*prefix_charge=nullptr,bool reference=false){
  size_t B=trial.size()/3;external(env,trial.data(),B,1,&sigma[i],&epsilon[i],&charge[i],energies,reference);
  for(size_t b=0;b<B;++b)energies[b]+=internal(prefix,get(trial,b),i,reference);
  if(reference)return;
  poremc::ChargeContribution rebuilt;
  if(longrange&&!prefix_charge){rebuilt=longrange->contribution(i,prefix.data(),charge.data());prefix_charge=&rebuilt;}
  reciprocal_batch(env,B,1,trial.data(),&charge[i],prefix_charge,energies);
 }
 void molecule_energies(void*env,const double*xyz,size_t batches,std::vector<double>&energies,bool include_longrange=true){
  external(env,xyz,batches,A,sigma.data(),epsilon.data(),charge.data(),energies);
  if(!rigid||longrange){
   std::vector<double>x(3*A);
   for(size_t b=0;b<batches;++b){std::copy(xyz+3*A*b,xyz+3*A*(b+1),x.begin());for(size_t i=0;i<A;++i)energies[b]+=internal(x,get(x,i),i);}
  }
  if(include_longrange)reciprocal_batch(env,batches,A,xyz,charge.data(),nullptr,energies);
 }
 double molecule_energy(void*env,const std::vector<double>&x,bool include_longrange=true){
  molecule_energies(env,x.data(),1,scratch_values,include_longrange);return scratch_values[0];
 }
 void surrogate_molecule_energies(void*env,const double*xyz,size_t batches,std::vector<double>&energies){
  external(env,xyz,batches,A,sigma.data(),epsilon.data(),charge.data(),energies,true);
  if(!rigid||longrange){
   std::vector<double>x(3*A);
   for(size_t b=0;b<batches;++b){std::copy(xyz+3*A*b,xyz+3*A*(b+1),x.begin());for(size_t i=0;i<A;++i)energies[b]+=internal(x,get(x,i),i,true);}
  }
 }
 double surrogate_molecule_energy(void*env,const std::vector<double>&x){
  surrogate_molecule_energies(env,x.data(),1,scratch_values);return scratch_values[0];
 }
 double host_internal_energy(const std::vector<double>&x){
  double u=0;if(poremc_query_energy(host_query.get(),1,x.data(),&u))throw std::runtime_error(poremc_last_error());
  if(!rigid||longrange)for(size_t i=0;i<A;++i)u+=internal(x,get(x,i),i);
  return u;
 }
 double reference_host_internal_energy(const std::vector<double>&x){
  double u=0;host_grid->batch_energy(1,A,x.data(),0,&u);
  if(!rigid||longrange)for(size_t i=0;i<A;++i)u+=internal(x,get(x,i),i,true);
  return u;
 }
 double selected_residual(void*env,const std::vector<double>&x,
                              const poremc::ChargeContribution*known,
                              poremc::ChargeContribution*prepared,
                              const double*cached_local,double*prepared_local,
                              const double*cached_reference,double*prepared_reference){
  // Guest-guest LJ and Ewald-real are identical in U and U0: cancel them
  // algebraically, rather than evaluating them a second time in stage 2.
  if(!cached_local){*prepared_local=host_internal_energy(x);cached_local=prepared_local;}
  if(!cached_reference){*prepared_reference=reference_host_internal_energy(x);cached_reference=prepared_reference;}
  double u=*cached_local-*cached_reference;
  if(longrange){
   if(!known){*prepared=longrange->contribution(A,x.data(),charge.data());known=prepared;}
   u+=longrange->insertion_energy(*known,excluded_charge(env));
  }
  return u;
 }
 std::pair<double,size_t> log_weight(const std::vector<double>&u,bool choose){double maximum=-std::numeric_limits<double>::infinity();for(double e:u){if(std::isnan(e)||e==-std::numeric_limits<double>::infinity())throw std::runtime_error("nonfinite negative or NaN CBMC energy");maximum=std::max(maximum,-e/T);}if(maximum==-std::numeric_limits<double>::infinity())return {maximum,0};if(!std::isfinite(maximum))throw std::runtime_error("CBMC log weight overflow");double sum=0;scratch_weights.resize(u.size());for(size_t j=0;j<u.size();++j){scratch_weights[j]=std::exp(-u[j]/T-maximum);sum+=scratch_weights[j];}size_t selected=0;if(choose){double target=uniform()*sum,cumulative=0;selected=u.size()-1;for(size_t i=0;i<u.size();++i){cumulative+=scratch_weights[i];if(target<cumulative){selected=i;break;}}}return {maximum+std::log(sum)-std::log(static_cast<double>(u.size())),selected};}
 Growth grow(void*env,const std::vector<double>*old=nullptr,bool reference=false){
  Growth result;result.x.resize(3*A);
  if(rigid){
   size_t k=K[0];std::vector<double>trials;trials.reserve(k*3*A);
   for(size_t j=0;j<k;++j){auto x=(old&&j==0)?*old:propose_rigid();trials.insert(trials.end(),x.begin(),x.end());}
   std::vector<double>energies;if(reference)surrogate_molecule_energies(env,trials.data(),k,energies);else molecule_energies(env,trials.data(),k,energies);
   if(old&&!std::isfinite(energies[0]))throw std::runtime_error("nonfinite actual old rigid retrace energy");
   if(!root_probability.empty())for(size_t j=0;j<k;++j)energies[j]+=T*root_logq({trials[j*3*A],trials[j*3*A+1],trials[j*3*A+2]});
   auto w=log_weight(energies,!old);result.logw=w.first;result.valid=std::isfinite(w.first);
   if(result.valid){size_t selected=old?0:w.second;std::copy(trials.begin()+selected*3*A,trials.begin()+(selected+1)*3*A,result.x.begin());result.energy=energies[selected]-(root_probability.empty()?0.:T*root_logq(get(result.x,0)));}
   return result;
  }
  // Prefixes are temporary, including their charge and reciprocal contribution.
  // A rejected trial cannot change the accepted structure factor / PME mesh.
  poremc::ChargeContribution prefix_charge;
  for(size_t i=0;i<A;++i){
   const auto&prefix=old?*old:result.x;auto&trial=scratch_trial;
   propose_batch(prefix,i,old,trial);auto&energies=scratch_energy;
   site_energies(env,prefix,i,trial,energies,longrange?&prefix_charge:nullptr,reference);
   if(old&&!std::isfinite(energies[0]))throw std::runtime_error("nonfinite actual old flexible retrace energy");
   if(i==0&&!root_probability.empty())for(size_t j=0;j<energies.size();++j)energies[j]+=T*root_logq(get(trial,j));
   auto w=log_weight(energies,!old);
   if(!std::isfinite(w.first)){result.valid=false;result.logw=-std::numeric_limits<double>::infinity();return result;}
   result.logw+=w.first;size_t selected=old?0:w.second;put(result.x,i,get(trial,selected));
   result.energy+=energies[selected]-((i==0&&!root_probability.empty())?T*root_logq(get(trial,selected)):0.);
   if(longrange&&!reference){auto site=longrange->contribution(1,trial.data()+3*selected,&charge[i]);longrange->accumulate(prefix_charge,site);}
  }
  return result;
 }
 void commit_insert(std::vector<double> x,poremc::ChargeContribution*prepared=nullptr,const double*prepared_local=nullptr,const double*prepared_reference=nullptr){
  if(longrange){auto item=prepared?std::move(*prepared):longrange->contribution(A,x.data(),charge.data());longrange->commit(&item);particle_charge.push_back(std::move(item));}
  if(delayed)particle_host_internal.push_back(prepared_local?*prepared_local:host_internal_energy(x));
  if(delayed)particle_reference_local.push_back(prepared_reference?*prepared_reference:reference_host_internal_energy(x));
  particles.push_back(std::move(x));guest_dirty=true;
 }
 void commit_erase(size_t index){
  if(longrange){longrange->commit(nullptr,&particle_charge[index]);particle_charge.erase(particle_charge.begin()+index);}
  if(delayed)particle_host_internal.erase(particle_host_internal.begin()+index);
  if(delayed)particle_reference_local.erase(particle_reference_local.begin()+index);
  particles.erase(particles.begin()+index);guest_dirty=true;
 }
 void commit_replace(size_t index,std::vector<double> x,poremc::ChargeContribution*prepared=nullptr,const double*prepared_local=nullptr,const double*prepared_reference=nullptr){
  if(longrange){auto item=prepared?std::move(*prepared):longrange->contribution(A,x.data(),charge.data());longrange->commit(&item,&particle_charge[index]);particle_charge[index]=std::move(item);}
  if(delayed)particle_host_internal[index]=prepared_local?*prepared_local:host_internal_energy(x);
  if(delayed)particle_reference_local[index]=prepared_reference?*prepared_reference:reference_host_internal_energy(x);
  particles[index]=std::move(x);guest_dirty=true;
 }
 bool accept(double ratio){if(std::isnan(ratio))throw std::runtime_error("NaN acceptance ratio");return std::log(std::max(uniform(),std::numeric_limits<double>::min()))<std::min(0.,ratio);}
 std::vector<double> local_proposal(const std::vector<double>&old,size_t move){
  auto candidate=old;
  if(move==2){V d{(2*uniform()-1)*max_translation,(2*uniform()-1)*max_translation,(2*uniform()-1)*max_translation};for(size_t i=0;i<A;++i)put(candidate,i,add(get(old,i),d));}
  else {V center{0,0,0};for(size_t i=0;i<A;++i)center=add(center,get(old,i));center=mul(center,1./A);V axis=direction();double theta=(2*uniform()-1)*max_rotation,c=std::cos(theta),s=std::sin(theta);for(size_t i=0;i<A;++i){V v=sub(get(old,i),center);V rotated=add(add(mul(v,c),mul(cross(axis,v),s)),mul(axis,(1-c)*dot(axis,v)));put(candidate,i,add(center,rotated));}}
  return candidate;
 }
 bool step_delayed(size_t move){
  size_t n=particles.size();auto*stats=acceleration_counters.data()+4*move;
  auto stage1=[&](double ratio){++stats[0];if(!accept(ratio))return false;++stats[1];return true;};
  auto stage2=[&](double ratio){if(!accept(ratio))return false;++stats[3];return true;};
  poremc::ChargeContribution prepared;double prepared_local=0,prepared_reference=0;
  if(move==0){
   if(z==0)return false;
   if(n>=max_particles)throw std::runtime_error("FastGCMC max_particles reached; increase capacity, not a physical truncation");
   void*env=cached_environment();auto candidate=grow(env,nullptr,true);
   if(!candidate.valid)return false;
   double loga=std::log(z)+std::log(Vbox)-std::log(double(n+1))+candidate.logw+std::log(probability[1]/probability[0]);
   if(!stage1(loga))return false;
   wrap(candidate.x);++stats[2];
   double residual=selected_residual(env,candidate.x,nullptr,&prepared,nullptr,&prepared_local,nullptr,&prepared_reference);
   if(!stage2(-residual/T))return false;
   commit_insert(std::move(candidate.x),longrange?&prepared:nullptr,&prepared_local,&prepared_reference);return true;
  }
  if(n==0)return false;
  size_t index=std::min(n-1,size_t(uniform()*n));const auto&old=particles[index];
  void*env=cached_environment(index);
  const auto*old_charge=longrange?&particle_charge[index]:nullptr;
  if(move==1){
   auto retrace=grow(env,&old,true);if(!retrace.valid)throw std::runtime_error("invalid reference deletion retrace");
   double loga=z==0?std::numeric_limits<double>::infinity():std::log(double(n))-std::log(z)-std::log(Vbox)-retrace.logw+std::log(probability[0]/probability[1]);
   if(!stage1(loga))return false;
   ++stats[2];double residual=selected_residual(env,old,old_charge,&prepared,&particle_host_internal[index],&prepared_local,&particle_reference_local[index],&prepared_reference);
   if(!std::isfinite(residual))throw std::runtime_error("invalid exact current molecule residual");
   if(!stage2(residual/T))return false;
   commit_erase(index);return true;
  }
  std::vector<double> candidate;double old_ref=0,new_ref=0,loga=0;
  if(move==4){
   auto grown=grow(env,nullptr,true);if(!grown.valid)return false;
   auto retrace=grow(env,&old,true);if(!retrace.valid)throw std::runtime_error("invalid reference regrowth retrace");
   old_ref=retrace.energy;new_ref=grown.energy;loga=grown.logw-retrace.logw;candidate=std::move(grown.x);
  }else{
   if(move==3&&A==1)return false;
   candidate=local_proposal(old,move);
   old_ref=surrogate_molecule_energy(env,old);new_ref=surrogate_molecule_energy(env,candidate);
   if(!std::isfinite(old_ref))throw std::runtime_error("invalid reference current molecule energy");
   loga=-(new_ref-old_ref)/T;
  }
  if(!stage1(loga))return false;
  wrap(candidate);
  const poremc::ChargeContribution*new_charge=nullptr;
  if(move==2&&longrange&&longrange->config().method==0){
   V d=sub(get(candidate,0),get(old,0));
   prepared=longrange->translated_contribution(*old_charge,{d.x,d.y,d.z});new_charge=&prepared;
  }
  stats[2]+=2;
  double old_residual=selected_residual(env,old,old_charge,&prepared,&particle_host_internal[index],&prepared_local,&particle_reference_local[index],&prepared_reference);
  double new_residual=selected_residual(env,candidate,new_charge,&prepared,nullptr,&prepared_local,nullptr,&prepared_reference);
  if(!std::isfinite(old_residual))throw std::runtime_error("invalid exact current molecule residual");
  if(!stage2(-(new_residual-old_residual)/T))return false;
  commit_replace(index,std::move(candidate),longrange?&prepared:nullptr,&prepared_local,&prepared_reference);return true;
 }
 void step(){double sel=uniform(),sum=0;size_t move=4;for(size_t m=0;m<5;++m){sum+=probability[m];if(sel<sum){move=m;break;}}++steps;++counters[2*move];bool accepted=false;size_t n=particles.size();
  if(delayed){if(step_delayed(move))++counters[2*move+1];return;}
  if(move==0){if(z==0)return;if(n>=max_particles)throw std::runtime_error("FastGCMC max_particles reached; increase capacity, not a physical truncation");void* env=cached_environment();auto candidate=grow(env);if(!candidate.valid)return;double ratio=std::log(z)+std::log(Vbox)-std::log(static_cast<double>(n+1))+candidate.logw+std::log(probability[1]/probability[0]);accepted=accept(ratio);if(accepted){wrap(candidate.x);commit_insert(std::move(candidate.x));}}
  else {if(n==0)return;size_t index=std::min(n-1,static_cast<size_t>(uniform()*n));const auto&old=particles[index];void* env=cached_environment(index);
   if(move==1){auto retrace=grow(env,&old);if(!retrace.valid)throw std::runtime_error("cannot retrace invalid current molecule");double ratio=z==0?std::numeric_limits<double>::infinity():std::log(static_cast<double>(n))-std::log(z)-std::log(Vbox)-retrace.logw+std::log(probability[0]/probability[1]);accepted=accept(ratio);if(accepted)commit_erase(index);}
   else if(move==4){auto candidate=grow(env);if(!candidate.valid)return;auto retrace=grow(env,&old);if(!retrace.valid)throw std::runtime_error("invalid old regrowth path");accepted=accept(candidate.logw-retrace.logw);if(accepted){wrap(candidate.x);commit_replace(index,std::move(candidate.x));}}
   else {auto candidate=old;if(move==2){V d{(2*uniform()-1)*max_translation,(2*uniform()-1)*max_translation,(2*uniform()-1)*max_translation};for(size_t i=0;i<A;++i)put(candidate,i,add(get(old,i),d));}
    else {if(A==1)return;V center{0,0,0};for(size_t i=0;i<A;++i)center=add(center,get(old,i));center=mul(center,1./A);V axis=direction();double theta=(2*uniform()-1)*max_rotation,c=std::cos(theta),s=std::sin(theta);for(size_t i=0;i<A;++i){V v=sub(get(old,i),center);V rotated=add(add(mul(v,c),mul(cross(axis,v),s)),mul(axis,(1-c)*dot(axis,v)));put(candidate,i,add(center,rotated));}}
    double old_u=molecule_energy(env,old),new_u=molecule_energy(env,candidate);if(!std::isfinite(old_u))throw std::runtime_error("invalid current molecule energy");accepted=accept(-(new_u-old_u)/T);if(accepted){wrap(candidate);commit_replace(index,std::move(candidate));}
   }
  }if(accepted){++counters[2*move+1];guest_dirty=true;}
 }
 double total_energy(){double result=0;for(size_t i=0;i<particles.size();++i){auto env=environment(std::numeric_limits<size_t>::max(),i);result+=molecule_energy(env.get(),particles[i],false);}if(longrange)result+=longrange->total_host_relative_energy();return result;}
};
}
extern "C" {
int poremc_fast_set_root(void*p,int nx,int ny,int nz,const double*prob){try{static_cast<Fast*>(p)->set_root(nx,ny,nz,prob);return 0;}catch(const std::exception&e){fast_error=e.what();return -1;}}
int poremc_fast_profile(void*p,uint64_t*out){if(!p||!out)return -1;auto&s=*static_cast<Fast*>(p);out[0]=s.guest_builds;out[1]=s.guest_reuses;out[2]=s.steps;return 0;}
size_t poremc_fast_config_size(){return sizeof(PoreMCFastConfig);}
const char* poremc_fast_last_error(){return fast_error.c_str();}
void* poremc_fast_create(const PoreMCFastConfig*c){try{fast_error.clear();if(!c)throw std::invalid_argument("null config");return new Fast(*c);}catch(const std::exception&e){fast_error=e.what();return nullptr;}}
void poremc_fast_destroy(void*p){delete static_cast<Fast*>(p);}
int poremc_fast_add(void*p,const double*coordinates){try{auto&s=*static_cast<Fast*>(p);if(s.particles.size()>=s.max_particles)throw std::runtime_error("max_particles exceeded");std::vector<double>x(coordinates,coordinates+3*s.A);for(double v:x)if(!std::isfinite(v))throw std::invalid_argument("nonfinite positions");s.wrap(x);auto env=s.environment();if(!std::isfinite(s.molecule_energy(env.get(),x)))throw std::invalid_argument("initial molecule has nonfinite energy");s.commit_insert(std::move(x));return 0;}catch(const std::exception&e){fast_error=e.what();return -1;}}
int poremc_fast_add_raw(void*p,const double*coordinates){try{auto&s=*static_cast<Fast*>(p);if(s.particles.size()>=s.max_particles)throw std::runtime_error("max_particles exceeded");std::vector<double>x(coordinates,coordinates+3*s.A);for(double v:x)if(!std::isfinite(v))throw std::invalid_argument("nonfinite positions");auto env=s.environment();if(!std::isfinite(s.molecule_energy(env.get(),x)))throw std::invalid_argument("restored molecule has nonfinite energy");s.commit_insert(std::move(x));return 0;}catch(const std::exception&e){fast_error=e.what();return -1;}}
size_t poremc_fast_count(void*p){return static_cast<Fast*>(p)->particles.size();}
uint64_t poremc_fast_step_count(void*p){return static_cast<Fast*>(p)->steps;}
int poremc_fast_positions(void*p,double*out){auto&s=*static_cast<Fast*>(p);for(size_t i=0;i<s.particles.size();++i)std::copy(s.particles[i].begin(),s.particles[i].end(),out+i*3*s.A);return 0;}
int poremc_fast_counters(void*p,uint64_t*out){auto&s=*static_cast<Fast*>(p);std::copy(s.counters.begin(),s.counters.end(),out);return 0;}
int poremc_fast_run(void*p,uint64_t burn,uint64_t steps,uint64_t every,uint64_t*counts,double*energies,int record,double*timing){try{if(every==0)throw std::invalid_argument("sample_every is zero");auto&s=*static_cast<Fast*>(p);auto a=std::chrono::steady_clock::now();for(uint64_t i=0;i<burn;++i)s.step();auto b=std::chrono::steady_clock::now();uint64_t out=0;for(uint64_t i=0;i<steps;++i){s.step();if((i+1)%every==0){counts[out]=s.particles.size();if(record)energies[out]=s.total_energy();++out;}}auto c=std::chrono::steady_clock::now();timing[0]=std::chrono::duration<double>(c-b).count();timing[1]=std::chrono::duration<double>(b-a).count();return 0;}catch(const std::exception&e){fast_error=e.what();return -1;}}
int poremc_fast_total_energy(void*p,double*out){try{*out=static_cast<Fast*>(p)->total_energy();return 0;}catch(const std::exception&e){fast_error=e.what();return -1;}}
size_t poremc_fast_rng(void*p,char*buffer,size_t capacity){auto&s=*static_cast<Fast*>(p);std::ostringstream stream;stream<<s.rng;std::string state=stream.str();if(buffer&&capacity>=state.size()+1)std::memcpy(buffer,state.c_str(),state.size()+1);return state.size()+1;}
int poremc_fast_restore_bookkeeping(void*p,const char*state,uint64_t steps,const uint64_t*counters){try{auto&s=*static_cast<Fast*>(p);std::istringstream stream(state);if(!(stream>>s.rng))throw std::invalid_argument("invalid mt19937_64 checkpoint state");s.steps=steps;std::copy(counters,counters+10,s.counters.begin());return 0;}catch(const std::exception&e){fast_error=e.what();return -1;}}
size_t poremc_fast_longrange_state(void*p,char*buffer,size_t capacity){
 try{auto&s=*static_cast<Fast*>(p);std::string state="none";
  if(s.longrange){
   std::ostringstream out;out<<std::setprecision(17)<<"PoreMC-FastEwald-v3\n"<<std::quoted(s.longrange->serialize())<<'\n'<<s.particle_charge.size()<<'\n';
   for(const auto&v:s.particle_charge){
    out<<v.charge<<' '<<v.squared_charge<<' '<<v.modes.size()<<' '<<v.indices.size()<<'\n';
    for(const auto&z:v.modes)out<<z.real()<<' '<<z.imag()<<'\n';
    for(size_t j=0;j<v.indices.size();++j)out<<v.indices[j]<<' '<<v.values[j]<<'\n';
   }
   state=out.str();
  }
  if(buffer&&capacity>=state.size()+1)std::memcpy(buffer,state.c_str(),state.size()+1);return state.size()+1;
 }catch(const std::exception&e){fast_error=e.what();return 0;}
}
int poremc_fast_restore_longrange(void*p,const char*state){
 try{auto&s=*static_cast<Fast*>(p);if(!state)throw std::invalid_argument("null reciprocal checkpoint");
  if(s.longrange){
   std::istringstream in(state);std::string version;std::getline(in,version);
   if(version=="PoreMC-FastEwald-v3"){
    std::string core;size_t count=0;in>>std::quoted(core)>>count;
    if(count!=s.particle_charge.size())throw std::invalid_argument("reciprocal particle count mismatch");
    std::vector<poremc::ChargeContribution> items(count);
    for(size_t i=0;i<count;++i){
     auto&v=items[i];size_t modes=0,entries=0;in>>v.charge>>v.squared_charge>>modes>>entries;
     if(modes!=s.particle_charge[i].modes.size()||entries!=s.particle_charge[i].indices.size())throw std::invalid_argument("reciprocal contribution shape mismatch");
     v.modes.resize(modes);v.indices.resize(entries);v.values.resize(entries);
     for(auto&z:v.modes){double re=0,im=0;in>>re>>im;z={re,im};}
     for(size_t j=0;j<entries;++j)in>>v.indices[j]>>v.values[j];
    }
    if(!in)throw std::invalid_argument("truncated reciprocal particle checkpoint");
    s.longrange->restore(core);s.particle_charge=std::move(items);
   }else s.longrange->restore(state);
  }
  else if(std::string(state)!="none")throw std::invalid_argument("reciprocal state supplied to a finite-range engine");
  return 0;
 }catch(const std::exception&e){fast_error=e.what();return -1;}
}
size_t poremc_fast_longrange_info(void*p,char*buffer,size_t capacity){
 try{auto&s=*static_cast<Fast*>(p);std::ostringstream out;
  if(!s.longrange)out<<"{\"method\":\"none\",\"backend\":\"none\"}";
  else{auto st=s.longrange->stats();out<<"{\"method\":\""<<(s.coulomb==3?"pme":"ewald")<<"\",\"backend\":\""<<s.longrange->backend_name()<<"\""
    <<",\"host_builds\":"<<st.host_builds<<",\"contributions\":"<<st.contributions
    <<",\"candidate_batches\":"<<st.candidate_batches<<",\"candidates\":"<<st.candidates
    <<",\"accepted_commits\":"<<st.accepted_commits<<",\"fft_transforms\":"<<st.fft_transforms
    <<",\"gpu_batches\":"<<st.gpu_batches<<",\"modes\":"<<st.modes<<",\"full_modes\":"<<st.full_modes
    <<",\"half_spectrum\":"<<(s.longrange->config().half_spectrum?"true":"false")
    <<",\"k_cutoff\":"<<s.longrange->config().k_cutoff<<",\"host_grid_builds\":"<<st.host_grid_builds
    <<",\"host_field_evaluations\":"<<st.host_field_evaluations<<",\"mesh_points\":"<<st.mesh_points<<"}";
  }
  std::string info=out.str();if(buffer&&capacity>=info.size()+1)std::memcpy(buffer,info.c_str(),info.size()+1);return info.size()+1;
 }catch(const std::exception&e){fast_error=e.what();return 0;}
}
int poremc_fast_site_energies(void*p,size_t exclude,size_t i,const double*prefix,size_t batches,const double*trials,double*out){
 try{auto&s=*static_cast<Fast*>(p);if(i>=s.A)throw std::invalid_argument("prefix length must be below atom count");
  if(exclude!=std::numeric_limits<size_t>::max()&&exclude>=s.particles.size())throw std::invalid_argument("excluded molecule index is out of range");
  if(!batches)return 0;std::vector<double>pre;if(i)pre.assign(prefix,prefix+3*i);
  std::vector<double>xyz(trials,trials+3*batches),energies;
  s.site_energies(s.cached_environment(exclude),pre,i,xyz,energies);
  std::copy(energies.begin(),energies.end(),out);return 0;
 }catch(const std::exception&e){fast_error=e.what();return -1;}
}
int poremc_fast_molecule_energies(void*p,size_t exclude,size_t batches,const double*trials,double*out){
 try{auto&s=*static_cast<Fast*>(p);
  if(exclude!=std::numeric_limits<size_t>::max()&&exclude>=s.particles.size())throw std::invalid_argument("excluded molecule index is out of range");
  if(!batches)return 0;std::vector<double>energies;
  s.molecule_energies(s.cached_environment(exclude),trials,batches,energies);
  std::copy(energies.begin(),energies.end(),out);return 0;
 }catch(const std::exception&e){fast_error=e.what();return -1;}
}
int poremc_fast_surrogate_site_energies(void*p,size_t exclude,size_t i,const double*prefix,size_t batches,const double*trials,double*out){
 try{auto&s=*static_cast<Fast*>(p);if(!s.host_grid)throw std::invalid_argument("surrogate probes require sampling_mode=delayed");
  if(i>=s.A)throw std::invalid_argument("prefix length must be below atom count");
  if(exclude!=std::numeric_limits<size_t>::max()&&exclude>=s.particles.size())throw std::invalid_argument("excluded molecule index is out of range");
  if(!batches)return 0;std::vector<double>pre;if(i)pre.assign(prefix,prefix+3*i);
  std::vector<double>xyz(trials,trials+3*batches),energies;
  s.site_energies(s.cached_environment(exclude),pre,i,xyz,energies,nullptr,true);
  std::copy(energies.begin(),energies.end(),out);return 0;
 }catch(const std::exception&e){fast_error=e.what();return -1;}
}
int poremc_fast_surrogate_molecule_energies(void*p,size_t exclude,size_t batches,const double*trials,double*out){
 try{auto&s=*static_cast<Fast*>(p);if(!s.host_grid)throw std::invalid_argument("surrogate probes require sampling_mode=delayed");
  if(exclude!=std::numeric_limits<size_t>::max()&&exclude>=s.particles.size())throw std::invalid_argument("excluded molecule index is out of range");
  if(!batches)return 0;std::vector<double>energies;
  s.surrogate_molecule_energies(s.cached_environment(exclude),trials,batches,energies);
  std::copy(energies.begin(),energies.end(),out);return 0;
 }catch(const std::exception&e){fast_error=e.what();return -1;}
}
int poremc_fast_acceleration_counters(void*p,uint64_t*out){if(!p||!out)return -1;auto&s=*static_cast<Fast*>(p);std::copy(s.acceleration_counters.begin(),s.acceleration_counters.end(),out);return 0;}
int poremc_fast_restore_acceleration_counters(void*p,const uint64_t*in){if(!p||!in)return -1;auto&s=*static_cast<Fast*>(p);std::copy(in,in+20,s.acceleration_counters.begin());return 0;}
size_t poremc_fast_acceleration_info(void*p,char*buffer,size_t capacity){
 try{auto&s=*static_cast<Fast*>(p);std::ostringstream out;
  out<<"{\"sampling_mode\":\""<<(s.delayed?"delayed":"exact")<<"\",\"selected_correction_backend\":\""<<(s.delayed?"cpu":"not_used")<<"\",\"grid_root_bias\":"<<s.root_bias<<",\"stages\":{";
  const char*names[5]={"insert","delete","translate","rotate","regrow"};
  uint64_t evaluated=0,passed=0,exact=0,accepted=0;
  for(size_t m=0;m<5;++m){auto*c=s.acceleration_counters.data()+4*m;if(m)out<<',';
   out<<'\"'<<names[m]<<"\":{\"stage1_evaluated\":"<<c[0]<<",\"stage1_passed\":"<<c[1]<<",\"exact_evaluations\":"<<c[2]<<",\"stage2_accepted\":"<<c[3]<<'}';
   evaluated+=c[0];passed+=c[1];exact+=c[2];accepted+=c[3];
  }
  out<<"},\"stage1_evaluated\":"<<evaluated<<",\"stage1_passed\":"<<passed<<",\"exact_evaluations\":"<<exact<<",\"stage2_accepted\":"<<accepted;
  if(s.host_grid){auto st=s.host_grid->stats();out<<",\"host_grid\":{\"builds\":"<<st.builds<<",\"cache_loads\":"<<st.cache_loads<<",\"cache_writes\":"<<st.cache_writes
   <<",\"lookups\":"<<st.lookups<<",\"cubic_lookups\":"<<st.cubic_lookups<<",\"linear_lookups\":"<<st.linear_lookups<<",\"boltzmann_lookups\":"<<st.boltzmann_lookups<<",\"exact_fallbacks\":"<<st.exact_fallbacks
   <<",\"steep_interpolation\":\""<<(s.host_grid->config().steep_interpolation==1?"boltzmann":"linear")<<"\",\"reference_cap_K\":"<<s.host_grid->config().surrogate_cap
   <<",\"mesh_points\":"<<st.mesh_points<<",\"lj_fields\":"<<st.lj_fields<<",\"site_types\":"<<st.site_types<<",\"bytes\":"<<st.bytes<<",\"steep_cells\":"<<st.steep_cells<<"}";}
  out<<'}';auto info=out.str();if(buffer&&capacity>=info.size()+1)std::memcpy(buffer,info.c_str(),info.size()+1);return info.size()+1;
 }catch(const std::exception&e){fast_error=e.what();return 0;}
}
}
