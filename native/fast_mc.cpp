// Independently derived, full C++ Monte Carlo loop. No upstream code copied.
#include "fast_mc.h"
#include "ewald_state.h"
#include "host_grid.h"
#include "guest_cells.h"
#include "short_range_cuda.h"
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
struct MoleculePlan {
 size_t A,ntors;
 double z,scale14lj,scale14q;
 bool rigid,has_partial_charge;
 std::vector<double> sigma,epsilon,charge,bond,angle,body,brange,arange,bk,ak,tk,tp;
 std::vector<int32_t> parent,aref,dref,K,excluded,tidx,tn,graph;
 Env host_query{nullptr,poremc_query_destroy};
 std::vector<Env> host_sites;
 std::vector<std::vector<size_t>> torsions_at;
 std::vector<double> bond_cube_lo,bond_cube_span,angle_cos,angle_cos_lo,angle_cos_span;
 std::vector<double> root_probability,root_threshold;
 std::vector<size_t> root_alias;
 std::array<int,3> root_shape{};
 std::unique_ptr<poremc::HostGrid> host_grid;
 explicit MoleculePlan(const PoreMCFastConfig& c)
     :A(c.atoms),ntors(c.torsions),z(c.activity),scale14lj(c.scale14_lj),
      scale14q(c.scale14_coulomb),rigid(c.rigid) {
  if(!A||z<0||!std::isfinite(z))throw std::invalid_argument("invalid molecule size or activity");
  auto dv=[&](const double*p,size_t n){return n ? std::vector<double>(p,p+n) : std::vector<double>{};};auto iv=[&](const int32_t*p,size_t n){return n ? std::vector<int32_t>(p,p+n) : std::vector<int32_t>{};};
  sigma=dv(c.sigma,A);epsilon=dv(c.epsilon,A);charge=dv(c.charge,A);bond=dv(c.bond_lengths,A);angle=dv(c.angles,A);body=dv(c.rigid_positions,3*A);
  has_partial_charge=std::any_of(charge.begin(),charge.end(),[](double value){return value!=0.0;});
  brange=dv(c.bond_ranges,2*A);arange=dv(c.angle_ranges,2*A);bk=dv(c.bond_constants,A);ak=dv(c.angle_constants,A);
  parent=iv(c.parents,A);aref=iv(c.angle_refs,A);dref=iv(c.dihedral_refs,A);K=iv(c.trials,A);excluded=iv(c.exclusions,A*A);
  tidx=iv(c.torsion_indices,4*ntors);tn=iv(c.torsion_periodicity,ntors);tk=dv(c.torsion_k,ntors);tp=dv(c.torsion_phase,ntors);
  for(auto k:K)if(k<1)throw std::invalid_argument("invalid CBMC trial count");
  graph.assign(A*A,static_cast<int32_t>(A+1));for(size_t i=0;i<A;++i){graph[i*A+i]=0;if(parent[i]>=0){graph[i*A+parent[i]]=1;graph[parent[i]*A+i]=1;}}
  for(size_t k=0;k<A;++k)for(size_t i=0;i<A;++i)for(size_t j=0;j<A;++j)graph[i*A+j]=std::min(graph[i*A+j],graph[i*A+k]+graph[k*A+j]);
  host_query=Env(poremc_query_create(c.host,A,sigma.data(),epsilon.data(),charge.data()),poremc_query_destroy);
  if(!host_query)throw std::runtime_error(poremc_last_error());
  for(size_t i=0;i<A;++i){host_sites.emplace_back(poremc_query_create(c.host,1,&sigma[i],&epsilon[i],&charge[i]),poremc_query_destroy);if(!host_sites.back())throw std::runtime_error(poremc_last_error());}
  torsions_at.resize(A);for(size_t t=0;t<ntors;++t)torsions_at[*std::max_element(tidx.begin()+4*t,tidx.begin()+4*t+4)].push_back(t);
  bond_cube_lo.resize(A);bond_cube_span.resize(A);angle_cos.resize(A);angle_cos_lo.resize(A);angle_cos_span.resize(A);
  for(size_t i=0;i<A;++i){double lo=brange[2*i],hi=brange[2*i+1];bond_cube_lo[i]=lo*lo*lo;bond_cube_span[i]=hi*hi*hi-lo*lo*lo;angle_cos[i]=std::cos(angle[i]);angle_cos_lo[i]=std::cos(arange[2*i+1]);angle_cos_span[i]=std::cos(arange[2*i])-angle_cos_lo[i];}
 }
};
struct Fast {
 size_t max_particles;
 double T,rc,alpha,max_translation,max_rotation,Vbox;
 int shifted,coulomb,algorithm,threads;
 std::array<double,9> H,inv;
 std::array<double,5> probability;
 std::vector<MoleculePlan> molecules;
 std::vector<double> component_cdf;
 const MoleculePlan* mol=nullptr;
 std::vector<std::vector<double>> particles;
 std::vector<size_t> particle_type;
 std::vector<std::vector<size_t>> members;
 std::unique_ptr<poremc::EwaldState> longrange;
 bool reference_cbmc=false;
 bool delayed_acceptance=false;
 bool partial_regrowth=false;
 double root_bias=0;
 // Counters per move: stage-1 evaluation/pass, exact evaluation, stage-2 accept.
 std::array<uint64_t,20> acceleration_counters{};
 std::vector<poremc::ChargeContribution> particle_charge;
 std::vector<double> particle_host_internal, particle_reference_local;
 std::vector<double> scratch_reciprocal;
 std::mt19937_64 rng;
 std::array<uint64_t,10> counters{};
 uint64_t steps=0;
 void* host;
 size_t framework_site_count;
 Env short_range_cuda{nullptr,poremc_short_cuda_destroy};
 size_t short_range_min_batch=1;
 bool short_range_guests_dirty=true;
 std::vector<PoreMCCudaPairSite> short_range_guest_sites;
 std::vector<size_t> short_range_guest_offsets;
 uint64_t cuda_framework_batches=0,cuda_guest_batches=0,cuda_guest_uploads=0;
 Env guest_cache{nullptr,poremc_environment_destroy};
 std::unique_ptr<poremc::GuestCells> guest_cells;
 bool guest_dirty=true;
 size_t excluded_particle=std::numeric_limits<size_t>::max();
 size_t excluded_site_begin=0, excluded_site_end=0;
 std::vector<size_t> guest_offsets;
 std::uint64_t guest_builds=0,guest_reuses=0,guest_updates=0;
 std::vector<double> scratch_other,scratch_trial,scratch_energy,scratch_values,scratch_weights,scratch_local_pair;
 double dsf_value=0,dsf_slope=0;

 Fast(const PoreMCFastConfig* configs,size_t components,const double* weights,int local,
      size_t mesh_refresh,bool background,bool partial,const char* lj_prefix,
      const char* electrostatic_path)
     :max_particles(configs[0].max_particles),T(configs[0].temperature),rc(configs[0].cutoff),
      alpha(configs[0].alpha),max_translation(configs[0].max_translation),max_rotation(configs[0].max_rotation),
      shifted(configs[0].shifted),coulomb(configs[0].coulomb),algorithm(configs[0].algorithm),
      threads(configs[0].threads),members(components),partial_regrowth(partial),rng(configs[0].seed),host(configs[0].host),framework_site_count(configs[0].host_atoms) {
  const auto& c=configs[0];
  dsf_value=std::erfc(alpha*rc)/rc;dsf_slope=std::erfc(alpha*rc)/(rc*rc)+2*alpha/std::sqrt(PI)*std::exp(-alpha*alpha*rc*rc)/rc;
  if(max_particles==0||!host||!(T>0)||!std::isfinite(T))throw std::invalid_argument("invalid FastGCMC thermodynamics/configuration");
  if(coulomb<0||coulomb>3)throw std::invalid_argument("unknown native Coulomb method");
  std::copy(c.cell,c.cell+9,H.begin());
  const double *h=H.data();Vbox=h[0]*(h[4]*h[8]-h[5]*h[7])-h[1]*(h[3]*h[8]-h[5]*h[6])+h[2]*(h[3]*h[7]-h[4]*h[6]);
  if(!(Vbox>0))throw std::invalid_argument("invalid cell");
  inv={(h[4]*h[8]-h[5]*h[7])/Vbox,(h[2]*h[7]-h[1]*h[8])/Vbox,(h[1]*h[5]-h[2]*h[4])/Vbox,(h[5]*h[6]-h[3]*h[8])/Vbox,(h[0]*h[8]-h[2]*h[6])/Vbox,(h[2]*h[3]-h[0]*h[5])/Vbox,(h[3]*h[7]-h[4]*h[6])/Vbox,(h[1]*h[6]-h[0]*h[7])/Vbox,(h[0]*h[4]-h[1]*h[3])/Vbox};
  double sum=0;for(size_t m=0;m<5;++m){probability[m]=c.move_weights[m];if(probability[m]<0||!std::isfinite(probability[m]))throw std::invalid_argument("invalid move probability");sum+=probability[m];}
  if(!(sum>0)||bool(probability[0])!=bool(probability[1]))throw std::invalid_argument("insertion and deletion weights must both be positive or both zero");
  for(auto&p:probability)p/=sum;
  molecules.reserve(components);
  for(size_t i=0;i<components;++i)molecules.emplace_back(configs[i]);
  mol=&molecules[0];
  sum=0.;
  for(size_t i=0;i<components;++i){
   const double weight=weights?weights[i]:1.;
   if(!std::isfinite(weight)||weight<=0.)throw std::invalid_argument("invalid component probability");
   sum+=weight;
   component_cdf.push_back(sum);
  }
  if(!std::isfinite(sum))throw std::invalid_argument("component probability overflow");
  for(auto& w:component_cdf)w/=sum;
  if(local)guest_cells=std::make_unique<poremc::GuestCells>(H.data(),rc,std::min(coulomb,2),alpha,shifted);
  if(coulomb>=2){
   poremc::ReciprocalConfig cfg;cfg.cell=H;cfg.alpha=alpha;cfg.threads=threads;
   cfg.half_spectrum=c.ewald_half_spectrum;cfg.k_cutoff=c.ewald_k_cutoff;
   for(int d=0;d<3;++d){cfg.kmax[d]=c.ewald_kmax[d];cfg.mesh[d]=c.pme_mesh[d];}
   cfg.method=c.reciprocal_method;cfg.backend=c.reciprocal_backend;
   cfg.pme_refresh=mesh_refresh;cfg.background=background;
   if(cfg.method!=(coulomb==3?1:0))throw std::invalid_argument("Coulomb and reciprocal method disagree");
   if(c.cuda_library)cfg.cuda_library=c.cuda_library;
   longrange=std::make_unique<poremc::EwaldState>(cfg,c.host_atoms,c.host_positions,c.host_charge);
  }
  if(c.sampling_mode<0||c.sampling_mode>2)throw std::invalid_argument("unknown sampling mode");
  reference_cbmc=c.sampling_mode!=0;
  delayed_acceptance=c.sampling_mode==1;
  root_bias=c.grid_root_bias;
  if(reference_cbmc)for(size_t i=0;i<components;++i){
   const auto& input=configs[i];
   auto& plan=molecules[i];
   mol=&plan;
   poremc::HostGridConfig gc;gc.cell=H;gc.threads=threads;gc.surrogate_cap=c.host_grid_cap;
   gc.temperature=T;gc.steep_interpolation=c.host_grid_steep;
   for(int d=0;d<3;++d)gc.mesh[d]=c.host_grid_mesh[d];
   if(input.host_grid_cache)gc.cache_path=input.host_grid_cache;
   if(input.host_grid_key)gc.cache_key=input.host_grid_key;
   if(lj_prefix)gc.lj_prefix=lj_prefix;
   if(electrostatic_path)gc.electrostatic_path=electrostatic_path;
   poremc::HostReciprocalGridBuilder build;
   poremc::HostReciprocalEnergy exact;
   if(longrange){
    build=[this,gc](){return longrange->host_potential_grid(gc.mesh);};
    exact=[this](size_t n,const double*x,const double*q){return longrange->host_insertion_energy(n,x,q);};
   }
   plan.host_grid=std::make_unique<poremc::HostGrid>(gc,host,plan.A,plan.sigma.data(),plan.epsilon.data(),plan.charge.data(),build,exact);
   if(root_bias>0)build_grid_root(plan,root_bias);
  }
  mol=&molecules[0];
 }
 void build_grid_root(MoleculePlan& plan,double bias){
  if(!(bias>0&&bias<1))throw std::invalid_argument("grid_root_bias must be in [0,1)");
  auto m=plan.host_grid->config().mesh;size_t n=size_t(m[0])*m[1]*m[2];
  std::vector<double> xyz(3*n),u(n),p(n);size_t k=0;
  for(int i=0;i<m[0];++i)for(int j=0;j<m[1];++j)for(int l=0;l<m[2];++l){
   put(xyz,k++,row({(i+.5)/m[0],(j+.5)/m[1],(l+.5)/m[2]},H));
  }
  plan.host_grid->batch_energy(n,1,xyz.data(),0,u.data());
  double minimum=*std::min_element(u.begin(),u.end()),sum=0;
  for(size_t i=0;i<n;++i){p[i]=std::exp(-(u[i]-minimum)/T);sum+=p[i];}
  if(!(sum>0)||!std::isfinite(sum))throw std::runtime_error("invalid host-grid root proposal");
  for(double&v:p)v=(1-bias)/n+bias*v/sum;
  set_root(plan,m[0],m[1],m[2],p.data());
 }
 double uniform(){return std::generate_canonical<double,53>(rng);}
 void set_root(MoleculePlan& plan,int nx,int ny,int nz,const double*p) {
  if(steps!=0)throw std::invalid_argument("root proposal must be fixed before production");
  if(nx<1||ny<1||nz<1)throw std::invalid_argument("invalid proposal grid");
  size_t n=size_t(nx)*ny*nz;plan.root_shape={nx,ny,nz};plan.root_probability.assign(p,p+n);
  double total=0;for(double v:plan.root_probability){if(!(v>0)||!std::isfinite(v))throw std::invalid_argument("proposal must have full finite support");total+=v;}
  if(!std::isfinite(total))throw std::invalid_argument("proposal overflow");
  for(double&v:plan.root_probability)v/=total;
  std::vector<double> scaled(n);std::vector<size_t> small,large;plan.root_alias.resize(n);plan.root_threshold.assign(n,1.);
  for(size_t i=0;i<n;++i){scaled[i]=plan.root_probability[i]*n;plan.root_alias[i]=i;(scaled[i]<1?small:large).push_back(i);}
  while(!small.empty()&&!large.empty()){
   size_t a=small.back(),b=large.back();small.pop_back();large.pop_back();plan.root_threshold[a]=scaled[a];plan.root_alias[a]=b;scaled[b]-=1-scaled[a];(scaled[b]<1?small:large).push_back(b);
  }
 }
 V root_position(){
  if(mol->root_probability.empty())return row({uniform(),uniform(),uniform()},H);
  double u=uniform()*mol->root_probability.size();size_t i=std::min(mol->root_probability.size()-1,size_t(u));
  size_t k=u-i<mol->root_threshold[i]?i:mol->root_alias[i];
  int iz=k%mol->root_shape[2],iy=(k/mol->root_shape[2])%mol->root_shape[1],ix=k/(size_t(mol->root_shape[2])*mol->root_shape[1]);
  return row({(ix+uniform())/mol->root_shape[0],(iy+uniform())/mol->root_shape[1],(iz+uniform())/mol->root_shape[2]},H);
 }
 double root_logq(V x)const{
  if(mol->root_probability.empty())return 0.;V f=row(x,inv);double v[3]={f.x,f.y,f.z};int key[3];
  for(int d=0;d<3;++d){v[d]-=std::floor(v[d]);key[d]=std::min(mol->root_shape[d]-1,int(v[d]*mol->root_shape[d]));}
  size_t k=(size_t(key[0])*mol->root_shape[1]+key[1])*mol->root_shape[2]+key[2];return std::log(mol->root_probability.size()*mol->root_probability[k]);
 }
 void* cached_environment(size_t exclude=std::numeric_limits<size_t>::max()){
  excluded_particle=exclude;
  if(guest_cells){++guest_reuses;return nullptr;}
  if(guest_dirty || !guest_cache){
   guest_cache=environment();
   guest_offsets.assign(1,0);
   for(size_t type:particle_type)guest_offsets.push_back(guest_offsets.back()+molecules[type].A);
   guest_dirty=false;
   ++guest_builds;
  }
  else ++guest_reuses;
  excluded_site_begin=excluded_site_end=0;
  if(exclude<particles.size()){
   excluded_site_begin=guest_offsets[exclude];
   excluded_site_end=guest_offsets[exclude+1];
  }
  return guest_cache.get();
 }

 double normal(){double u=std::max(uniform(),std::numeric_limits<double>::min());return std::sqrt(-2*std::log(u))*std::cos(2*PI*uniform());}
 V direction(){double c=2*uniform()-1,p=2*PI*uniform(),s=std::sqrt(std::max(0.,1-c*c));return {s*std::cos(p),s*std::sin(p),c};}
 std::array<double,9> rotation(){double w=normal(),x=normal(),y=normal(),zz=normal(),n=std::sqrt(w*w+x*x+y*y+zz*zz);w/=n;x/=n;y/=n;zz/=n;return {1-2*(y*y+zz*zz),2*(x*y-w*zz),2*(x*zz+w*y),2*(x*y+w*zz),1-2*(x*x+zz*zz),2*(y*zz-w*x),2*(x*zz-w*y),2*(y*zz+w*x),1-2*(x*x+y*y)};}
 V apply_rotation(V x,const std::array<double,9>&r){return {r[0]*x.x+r[1]*x.y+r[2]*x.z,r[3]*x.x+r[4]*x.y+r[5]*x.z,r[6]*x.x+r[7]*x.y+r[8]*x.z};}
 void wrap(std::vector<double>& x){V f=row(get(x,0),inv);V d=row({std::floor(f.x),std::floor(f.y),std::floor(f.z)},H);for(size_t i=0;i<mol->A;++i)put(x,i,sub(get(x,i),d));}
 std::vector<double> propose_rigid(){V root=root_position();if(mol->A==1)return {root.x,root.y,root.z};auto rot=rotation();V origin=mol->root_probability.empty()?V{0,0,0}:apply_rotation(get(mol->body,0),rot);std::vector<double>x(3*mol->A);for(size_t i=0;i<mol->A;++i)put(x,i,add(sub(apply_rotation(get(mol->body,i),rot),origin),root));return x;}
 V propose(const std::vector<double>& prefix,size_t i){
  if(i==0)return root_position();
  double r=mol->bond[i];if(mol->brange[2*i+1]>mol->brange[2*i]){double lo=mol->brange[2*i],hi=mol->brange[2*i+1];r=std::cbrt(lo*lo*lo+uniform()*(hi*hi*hi-lo*lo*lo));}
  V p=get(prefix,mol->parent[i]);if(mol->aref[i]<0)return add(p,mul(direction(),r));
  V e1=unit(sub(get(prefix,mol->aref[i]),p));
  V v=mol->dref[i]>=0?sub(get(prefix,mol->dref[i]),get(prefix,mol->aref[i])):(std::fabs(e1.x)<=std::fabs(e1.y)&&std::fabs(e1.x)<=std::fabs(e1.z)?V{1,0,0}:(std::fabs(e1.y)<=std::fabs(e1.z)?V{0,1,0}:V{0,0,1}));
  V e2=sub(v,mul(e1,dot(v,e1)));
  if(norm(e2)<1e-12){v=std::fabs(e1.x)<=std::fabs(e1.y)&&std::fabs(e1.x)<=std::fabs(e1.z)?V{1,0,0}:(std::fabs(e1.y)<=std::fabs(e1.z)?V{0,1,0}:V{0,0,1});e2=sub(v,mul(e1,dot(v,e1)));}
  e2=unit(e2);V e3=cross(e1,e2);double cosine=std::cos(mol->angle[i]);
  if(mol->arange[2*i+1]>mol->arange[2*i])cosine=std::cos(mol->arange[2*i+1])+uniform()*(std::cos(mol->arange[2*i])-std::cos(mol->arange[2*i+1]));
  double phi=2*PI*uniform()-PI,sine=std::sqrt(std::max(0.,1-cosine*cosine));
  return add(p,mul(add(mul(e1,cosine),mul(add(mul(e2,std::cos(phi)),mul(e3,std::sin(phi))),sine)),r));
 }
 void propose_batch(const std::vector<double>&prefix,size_t i,const std::vector<double>*old,std::vector<double>&trial){
  trial.resize(3*mol->K[i]);size_t begin=0;if(old){put(trial,0,get(*old,i));begin=1;}
  V origin{0,0,0},e1{0,0,0},e2{0,0,0},e3{0,0,0};
  if(i){origin=get(prefix,mol->parent[i]);if(mol->aref[i]>=0){
   e1=unit(sub(get(prefix,mol->aref[i]),origin));
   V v=mol->dref[i]>=0?sub(get(prefix,mol->dref[i]),get(prefix,mol->aref[i])):(std::fabs(e1.x)<=std::fabs(e1.y)&&std::fabs(e1.x)<=std::fabs(e1.z)?V{1,0,0}:(std::fabs(e1.y)<=std::fabs(e1.z)?V{0,1,0}:V{0,0,1}));
   e2=sub(v,mul(e1,dot(v,e1)));if(norm(e2)<1e-12){v=std::fabs(e1.x)<=std::fabs(e1.y)&&std::fabs(e1.x)<=std::fabs(e1.z)?V{1,0,0}:(std::fabs(e1.y)<=std::fabs(e1.z)?V{0,1,0}:V{0,0,1});e2=sub(v,mul(e1,dot(v,e1)));}
   e2=unit(e2);e3=cross(e1,e2);
  }}
  for(size_t j=begin;j<size_t(mol->K[i]);++j){
   if(i==0){put(trial,j,root_position());continue;}
   double r=mol->bond[i];if(mol->brange[2*i+1]>mol->brange[2*i])r=std::cbrt(mol->bond_cube_lo[i]+uniform()*mol->bond_cube_span[i]);
   if(mol->aref[i]<0){put(trial,j,add(origin,mul(direction(),r)));continue;}
   double cosine=mol->angle_cos[i];if(mol->arange[2*i+1]>mol->arange[2*i])cosine=mol->angle_cos_lo[i]+uniform()*mol->angle_cos_span[i];
   double phi=2*PI*uniform()-PI,sine=std::sqrt(std::max(0.,1-cosine*cosine));
   put(trial,j,add(origin,mul(add(mul(e1,cosine),mul(add(mul(e2,std::cos(phi)),mul(e3,std::sin(phi))),sine)),r)));
  }
 }
 Env environment(size_t exclude=std::numeric_limits<size_t>::max(),size_t limit=std::numeric_limits<size_t>::max()){
  std::vector<double>p,s,e,q;
  const size_t used=std::min(particles.size(),limit);
  size_t sites=0;
  for(size_t i=0;i<used;++i)if(i!=exclude)sites+=molecules[particle_type[i]].A;
  p.reserve(sites*3);s.reserve(sites);e.reserve(sites);q.reserve(sites);
  for(size_t i=0;i<used;++i)if(i!=exclude){
   const auto& g=molecules[particle_type[i]];
   p.insert(p.end(),particles[i].begin(),particles[i].end());
   s.insert(s.end(),g.sigma.begin(),g.sigma.end());
   e.insert(e.end(),g.epsilon.begin(),g.epsilon.end());
   q.insert(q.end(),g.charge.begin(),g.charge.end());
  }
  void* handle=poremc_environment_create(s.size(),p.data(),s.data(),e.data(),q.data(),H.data(),rc,std::min(coulomb,2),alpha,shifted,algorithm==2?1:algorithm,threads);
  if(!handle)throw std::runtime_error(poremc_last_error());return Env(handle,poremc_environment_destroy);
 }
 bool use_cuda_short_range(size_t batches) const {
  return short_range_cuda && batches>=short_range_min_batch;
 }
 void upload_cuda_guests(){
  if(!short_range_guests_dirty)return;
  short_range_guest_sites.clear();
  short_range_guest_offsets.clear();
  short_range_guest_offsets.push_back(0);
  for(size_t i=0;i<particles.size();++i){
   const auto& plan=molecules[particle_type[i]];
   for(size_t a=0;a<plan.A;++a){
    const auto f=row(get(particles[i],a),inv);
    short_range_guest_sites.push_back({{f.x-std::floor(f.x),f.y-std::floor(f.y),f.z-std::floor(f.z)},
                                      plan.sigma[a],std::sqrt(plan.epsilon[a]),plan.charge[a]});
   }
   short_range_guest_offsets.push_back(short_range_guest_sites.size());
  }
  if(poremc_short_cuda_upload_guests(short_range_cuda.get(),short_range_guest_sites.size(),
                                    short_range_guest_sites.data()))
   throw std::runtime_error(poremc_last_error());
  short_range_guests_dirty=false;
  ++cuda_guest_uploads;
 }
 void cuda_pairs(size_t batches,size_t atoms,const double* positions,
                 const double* sigma,const double* epsilon,const double* charge,int parts,double* out){
  size_t begin=0,end=0;
  if(parts&POREMC_PAIR_GUESTS){
   upload_cuda_guests();
   if(excluded_particle<particles.size()){
    begin=short_range_guest_offsets[excluded_particle];
    end=short_range_guest_offsets[excluded_particle+1];
   }
  }
  if(poremc_short_cuda_energy(short_range_cuda.get(),batches,atoms,positions,sigma,epsilon,charge,begin,end,parts,out))
   throw std::runtime_error(poremc_last_error());
  if(parts&POREMC_PAIR_FRAMEWORK)++cuda_framework_batches;
  if(parts&POREMC_PAIR_GUESTS)++cuda_guest_batches;
 }
 void external(void* env,const double* positions,size_t batches,size_t atoms,const double*s,const double*e,const double*q,std::vector<double>&out,bool reference=false){
  out.resize(batches);
  const bool grid=reference&&mol->host_grid;
  // Arbitrary partial environments (energy reconstruction) keep their own
  // CPU path. The device snapshot mirrors the complete accepted guest state.
  const bool gpu=use_cuda_short_range(batches)&&(!env||env==guest_cache.get());
  if(gpu&&!grid&&(framework_site_count||!particles.empty())){
   const int sides=(framework_site_count?POREMC_PAIR_FRAMEWORK:0)
                   | (particles.empty()?0:POREMC_PAIR_GUESTS);
   cuda_pairs(batches,atoms,positions,s,e,q,sides,out.data());
   return;
  }
  void* binding=atoms==mol->A?mol->host_query.get():mol->host_sites[size_t(s-mol->sigma.data())].get();
  if(grid)mol->host_grid->batch_energy(batches,atoms,positions,size_t(s-mol->sigma.data()),out.data());
  else if(poremc_query_energy(binding,batches,positions,out.data()))throw std::runtime_error(poremc_last_error());
  scratch_other.resize(batches);auto&other=scratch_other;
  if(gpu&&!particles.empty()){
   cuda_pairs(batches,atoms,positions,s,e,q,POREMC_PAIR_GUESTS,other.data());
  }else if(!env){
   const size_t end=excluded_particle<particles.size()?excluded_particle+1:excluded_particle;
   guest_cells->batch(batches,atoms,positions,s,e,q,excluded_particle,end,other.data());
  }else{
   const bool current=env==guest_cache.get();
   if(poremc_environment_energy_excluding(env,batches,atoms,positions,s,e,q,
       current?excluded_site_begin:0,current?excluded_site_end:0,other.data()))
    throw std::runtime_error(poremc_last_error());
  }
  for(size_t b=0;b<batches;++b)out[b]+=other[b];
 }
 double internal(const std::vector<double>&prefix,V x,size_t i,bool reference=false){
  if(mol->rigid&&!longrange)return 0.;double u=0.;
  for(size_t j=0;j<i;++j){double r=norm(sub(x,get(prefix,j)));bool ex=mol->rigid||mol->excluded[i*mol->A+j];double lj=ex?0.:(mol->graph[i*mol->A+j]==3?mol->scale14lj:1.);double cq=ex?0.:(mol->graph[i*mol->A+j]==3?mol->scale14q:1.);
   if(r<rc){double s=.5*(mol->sigma[i]+mol->sigma[j]),e=std::sqrt(mol->epsilon[i]*mol->epsilon[j]);
    if(lj&&s&&e){
     if(r<=1e-12)return std::numeric_limits<double>::infinity();
     const double t6=std::pow(s/r,6);
     double v=4*e*t6*(t6-1);
     if(shifted){
      const double c6=std::pow(s/rc,6);
      v-=4*e*c6*(c6-1);
      if(shifted==2)v-=24*e*c6*(1-2*c6)*(r-rc)/rc;
     }
     u+=lj*v;
    }
    double qq=mol->charge[i]*mol->charge[j];if(coulomb==1&&cq&&qq){if(r<=1e-12)return std::numeric_limits<double>::infinity();u+=cq*CK*qq*(std::erfc(alpha*r)/r-dsf_value+dsf_slope*(r-rc));}
   }
   if(longrange&&mol->charge[i]*mol->charge[j]!=0.){
    // Ewald includes every mol->charge in reciprocal space. Remove the *bare*
    // primary interaction for exclusions / 1-4 scaling, retaining images.
    // Molecular coordinates are unwrapped, never minimum-imaged here.
    double radial=0.;
    if(r<=1e-12){if(cq!=0.)return std::numeric_limits<double>::infinity();radial=reference?0.:-2*alpha/std::sqrt(PI);}
    else if(r<rc)radial=reference?cq/r:(cq==0.?-std::erf(alpha*r):std::erfc(alpha*r)-(1-cq))/r;
    else radial=reference?(std::erf(alpha*r)-(1-cq))/r:-(1-cq)/r;
    u+=CK*mol->charge[i]*mol->charge[j]*radial;
   }
  }
  if(mol->rigid)return u;
  if(i){double r=norm(sub(x,get(prefix,mol->parent[i])));u+=.5*mol->bk[i]*(r-mol->bond[i])*(r-mol->bond[i]);if(mol->aref[i]>=0&&mol->ak[i]){V a=sub(x,get(prefix,mol->parent[i])),b=sub(get(prefix,mol->aref[i]),get(prefix,mol->parent[i]));double co=std::clamp(dot(a,b)/(norm(a)*norm(b)),-1.,1.);double delta=std::acos(co)-mol->angle[i];u+=.5*mol->ak[i]*delta*delta;}}
  for(size_t t:mol->torsions_at[i]){V p[4];for(size_t k=0;k<4;++k)p[k]=mol->tidx[4*t+k]==static_cast<int>(i)?x:get(prefix,mol->tidx[4*t+k]);V b0=sub(p[0],p[1]),b1=unit(sub(p[2],p[1])),b2=sub(p[3],p[2]);V v=sub(b0,mul(b1,dot(b0,b1))),w=sub(b2,mul(b1,dot(b2,b1)));double phi=std::atan2(dot(cross(b1,v),w),dot(v,w));u+=mol->tk[t]*(1+std::cos(mol->tn[t]*phi-mol->tp[t]));}
  return u;
 }
 const poremc::ChargeContribution* excluded_charge(void*env)const{
  return longrange&&(!env||env==guest_cache.get())&&excluded_particle<particle_charge.size()?&particle_charge[excluded_particle]:nullptr;
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
  size_t B=trial.size()/3;external(env,trial.data(),B,1,&mol->sigma[i],&mol->epsilon[i],&mol->charge[i],energies,reference);
  for(size_t b=0;b<B;++b)energies[b]+=internal(prefix,get(trial,b),i,reference);
  if(reference)return;
  poremc::ChargeContribution rebuilt;
  if(longrange&&!prefix_charge){rebuilt=longrange->contribution(i,prefix.data(),mol->charge.data());prefix_charge=&rebuilt;}
  reciprocal_batch(env,B,1,trial.data(),&mol->charge[i],prefix_charge,energies);
 }
 void molecule_energies(void*env,const double*xyz,size_t batches,std::vector<double>&energies,bool include_longrange=true){
  external(env,xyz,batches,mol->A,mol->sigma.data(),mol->epsilon.data(),mol->charge.data(),energies);
  if(!mol->rigid||(longrange&&mol->has_partial_charge)){
   std::vector<double>x(3*mol->A);
   for(size_t b=0;b<batches;++b){std::copy(xyz+3*mol->A*b,xyz+3*mol->A*(b+1),x.begin());for(size_t i=0;i<mol->A;++i)energies[b]+=internal(x,get(x,i),i);}
  }
  if(include_longrange)reciprocal_batch(env,batches,mol->A,xyz,mol->charge.data(),nullptr,energies);
 }
 double molecule_energy(void*env,const std::vector<double>&x,bool include_longrange=true){
  molecule_energies(env,x.data(),1,scratch_values,include_longrange);return scratch_values[0];
 }
 void surrogate_molecule_energies(void*env,const double*xyz,size_t batches,std::vector<double>&energies){
  external(env,xyz,batches,mol->A,mol->sigma.data(),mol->epsilon.data(),mol->charge.data(),energies,true);
  if(!mol->rigid||longrange){
   std::vector<double>x(3*mol->A);
   for(size_t b=0;b<batches;++b){std::copy(xyz+3*mol->A*b,xyz+3*mol->A*(b+1),x.begin());for(size_t i=0;i<mol->A;++i)energies[b]+=internal(x,get(x,i),i,true);}
  }
 }
 double surrogate_molecule_energy(void*env,const std::vector<double>&x){
  surrogate_molecule_energies(env,x.data(),1,scratch_values);return scratch_values[0];
 }
 double host_internal_energy(const std::vector<double>&x,bool allow_cuda=true){
  double u=0;
  if(allow_cuda&&use_cuda_short_range(1)&&framework_site_count)
   cuda_pairs(1,mol->A,x.data(),mol->sigma.data(),mol->epsilon.data(),mol->charge.data(),POREMC_PAIR_FRAMEWORK,&u);
  else if(poremc_query_energy(mol->host_query.get(),1,x.data(),&u))
   throw std::runtime_error(poremc_last_error());
  if(!mol->rigid||longrange)for(size_t i=0;i<mol->A;++i)u+=internal(x,get(x,i),i);
  return u;
 }
 double reference_host_internal_energy(const std::vector<double>&x){
  double u=0;mol->host_grid->batch_energy(1,mol->A,x.data(),0,&u);
  if(!mol->rigid||longrange)for(size_t i=0;i<mol->A;++i)u+=internal(x,get(x,i),i,true);
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
   if(!known){*prepared=longrange->selected_contribution(mol->A,x.data(),mol->charge.data());known=prepared;}
   u+=longrange->insertion_energy(*known,excluded_charge(env));
  }
  return u;
 }
 std::pair<double,size_t> log_weight(const std::vector<double>&u,bool choose){double maximum=-std::numeric_limits<double>::infinity();for(double e:u){if(std::isnan(e)||e==-std::numeric_limits<double>::infinity())throw std::runtime_error("nonfinite negative or NaN CBMC energy");maximum=std::max(maximum,-e/T);}if(maximum==-std::numeric_limits<double>::infinity())return {maximum,0};if(!std::isfinite(maximum))throw std::runtime_error("CBMC log weight overflow");double sum=0;scratch_weights.resize(u.size());for(size_t j=0;j<u.size();++j){scratch_weights[j]=std::exp(-u[j]/T-maximum);sum+=scratch_weights[j];}size_t selected=0;if(choose){double target=uniform()*sum,cumulative=0;selected=u.size()-1;for(size_t i=0;i<u.size();++i){cumulative+=scratch_weights[i];if(target<cumulative){selected=i;break;}}}return {maximum+std::log(sum)-std::log(static_cast<double>(u.size())),selected};}
 Growth grow(void*env,const std::vector<double>*old=nullptr,bool reference=false,
             size_t start=0,const std::vector<double>*initial=nullptr){
  Growth result;result.x.resize(3*mol->A);
  if(mol->rigid){
   size_t k=mol->K[0];std::vector<double>trials;trials.reserve(k*3*mol->A);
   for(size_t j=0;j<k;++j){auto x=(old&&j==0)?*old:propose_rigid();trials.insert(trials.end(),x.begin(),x.end());}
   std::vector<double>energies;if(reference)surrogate_molecule_energies(env,trials.data(),k,energies);else molecule_energies(env,trials.data(),k,energies);
   if(old&&!std::isfinite(energies[0]))throw std::runtime_error("nonfinite actual old rigid retrace energy");
   if(!mol->root_probability.empty())for(size_t j=0;j<k;++j)energies[j]+=T*root_logq({trials[j*3*mol->A],trials[j*3*mol->A+1],trials[j*3*mol->A+2]});
   auto w=log_weight(energies,!old);result.logw=w.first;result.valid=std::isfinite(w.first);
   if(result.valid){size_t selected=old?0:w.second;std::copy(trials.begin()+selected*3*mol->A,trials.begin()+(selected+1)*3*mol->A,result.x.begin());result.energy=energies[selected]-(mol->root_probability.empty()?0.:T*root_logq(get(result.x,0)));}
   return result;
  }
  // Prefixes are temporary, including their charge and reciprocal contribution.
  // mol->A rejected trial cannot change the accepted structure factor / PME mesh.
  poremc::ChargeContribution prefix_charge;
  if(start){
   const auto& fixed=old?*old:*initial;
   std::copy_n(fixed.begin(),3*start,result.x.begin());
   if(longrange&&!reference)prefix_charge=longrange->contribution(start,fixed.data(),mol->charge.data());
  }
  for(size_t i=start;i<mol->A;++i){
   const auto&prefix=old?*old:result.x;auto&trial=scratch_trial;
   propose_batch(prefix,i,old,trial);auto&energies=scratch_energy;
   site_energies(env,prefix,i,trial,energies,longrange?&prefix_charge:nullptr,reference);
   if(old&&!std::isfinite(energies[0]))throw std::runtime_error("nonfinite actual old flexible retrace energy");
   if(i==0&&!mol->root_probability.empty())for(size_t j=0;j<energies.size();++j)energies[j]+=T*root_logq(get(trial,j));
   auto w=log_weight(energies,!old);
   if(!std::isfinite(w.first)){result.valid=false;result.logw=-std::numeric_limits<double>::infinity();return result;}
   result.logw+=w.first;size_t selected=old?0:w.second;put(result.x,i,get(trial,selected));
   result.energy+=energies[selected]-((i==0&&!mol->root_probability.empty())?T*root_logq(get(trial,selected)):0.);
   if(longrange&&!reference){auto site=longrange->contribution(1,trial.data()+3*selected,&mol->charge[i]);longrange->accumulate(prefix_charge,site);}
  }
  return result;
 }
 size_t regrowth_start(){
  return partial_regrowth&&!mol->rigid?std::min(mol->A-1,size_t(uniform()*mol->A)):0;
 }
 size_t component()const{return size_t(mol-molecules.data());}
 void select(size_t species){
  if(species>=molecules.size())throw std::invalid_argument("invalid component index");
  mol=&molecules[species];
 }
 template<class V> static void reserve_one(V& v){
  if(v.size()==v.capacity())v.reserve(v.size()?2*v.size():1);
 }
 void insert_cells(size_t index,const MoleculePlan& plan,const std::vector<double>& x){
  guest_cells->insert(index,plan.A,x.data(),plan.sigma.data(),plan.epsilon.data(),plan.charge.data());
  ++guest_updates;
 }
 void commit_insert(std::vector<double> x,poremc::ChargeContribution*prepared=nullptr,const double*prepared_local=nullptr,const double*prepared_reference=nullptr){
  auto& group=members[component()];
  reserve_one(group);reserve_one(particle_type);reserve_one(particles);
  auto item=longrange?(prepared?std::move(*prepared):longrange->contribution(mol->A,x.data(),mol->charge.data())):poremc::ChargeContribution{};
  if(longrange)reserve_one(particle_charge);
  double local=0,reference=0;
  if(reference_cbmc){
   local=prepared_local?*prepared_local:host_internal_energy(x);
   reference=prepared_reference?*prepared_reference:reference_host_internal_energy(x);
   reserve_one(particle_host_internal);reserve_one(particle_reference_local);
  }
  if(guest_cells)guest_cells->reserve(mol->A,x.data());
  if(longrange){longrange->commit(&item);particle_charge.push_back(std::move(item));}
  if(guest_cells)insert_cells(particles.size(),*mol,x);
  if(reference_cbmc){particle_host_internal.push_back(local);particle_reference_local.push_back(reference);}
  group.push_back(particles.size());
  particle_type.push_back(component());
  particles.push_back(std::move(x));guest_dirty=short_range_guests_dirty=true;
 }
 template<class V> static void swap_erase(V& v,size_t i){
  if(i+1!=v.size())v[i]=std::move(v.back());
  v.pop_back();
 }
 void commit_erase(size_t index){
  const size_t last=particles.size()-1, tail_type=particle_type.back();
  const auto& tail=molecules[tail_type];
  if(guest_cells&&index!=last)guest_cells->reserve(tail.A,particles.back().data());
  if(longrange){longrange->commit(nullptr,&particle_charge[index]);swap_erase(particle_charge,index);}
  if(reference_cbmc){swap_erase(particle_host_internal,index);swap_erase(particle_reference_local,index);}
  if(guest_cells){
   guest_cells->erase(index,mol->A,particles[index].data());
   if(index!=last){
    guest_cells->erase(last,tail.A,particles.back().data());
    insert_cells(index,tail,particles.back());
   }else ++guest_updates;
  }
  auto& selected=members[component()];
  selected.erase(std::lower_bound(selected.begin(),selected.end(),index));
  if(index!=last){
   auto& other=members[tail_type];
   other.pop_back(); // last is the greatest member index
   other.insert(std::lower_bound(other.begin(),other.end(),index),index);
  }
  swap_erase(particle_type,index);swap_erase(particles,index);
  guest_dirty=short_range_guests_dirty=true;
 }
 void commit_replace(size_t index,std::vector<double> x,poremc::ChargeContribution*prepared=nullptr,const double*prepared_local=nullptr,const double*prepared_reference=nullptr){
  const double local=reference_cbmc?(prepared_local?*prepared_local:host_internal_energy(x)):0.;
  const double reference=reference_cbmc?(prepared_reference?*prepared_reference:reference_host_internal_energy(x)):0.;
  if(guest_cells)guest_cells->reserve(mol->A,x.data());
  if(longrange){
   auto item=prepared?std::move(*prepared):longrange->contribution(mol->A,x.data(),mol->charge.data());
   longrange->commit(&item,&particle_charge[index]);particle_charge[index]=std::move(item);
  }
  if(guest_cells){guest_cells->erase(index,mol->A,particles[index].data());insert_cells(index,*mol,x);}
  if(reference_cbmc){particle_host_internal[index]=local;particle_reference_local[index]=reference;}
  particles[index]=std::move(x);guest_dirty=short_range_guests_dirty=true;
 }
 void add_particle(size_t species,const double* coordinates,bool raw){
  select(species);
  if(particles.size()>=max_particles)throw std::runtime_error("max_particles exceeded");
  std::vector<double> x(coordinates,coordinates+3*mol->A);
  for(double v:x)if(!std::isfinite(v))throw std::invalid_argument("nonfinite positions");
  if(!raw)wrap(x);
  if(!std::isfinite(molecule_energy(cached_environment(),x)))
   throw std::invalid_argument("initial molecule has nonfinite energy");
  commit_insert(std::move(x));
 }
 bool accept(double ratio){if(std::isnan(ratio))throw std::runtime_error("NaN acceptance ratio");return std::log(std::max(uniform(),std::numeric_limits<double>::min()))<std::min(0.,ratio);}
 std::vector<double> local_proposal(const std::vector<double>&old,size_t move){
  auto candidate=old;
  if(move==2){V d{(2*uniform()-1)*max_translation,(2*uniform()-1)*max_translation,(2*uniform()-1)*max_translation};for(size_t i=0;i<mol->A;++i)put(candidate,i,add(get(old,i),d));}
  else {V center{0,0,0};for(size_t i=0;i<mol->A;++i)center=add(center,get(old,i));center=mul(center,1./mol->A);V axis=direction();double theta=(2*uniform()-1)*max_rotation,c=std::cos(theta),s=std::sin(theta);for(size_t i=0;i<mol->A;++i){V v=sub(get(old,i),center);V rotated=add(add(mul(v,c),mul(cross(axis,v),s)),mul(axis,(1-c)*dot(axis,v)));put(candidate,i,add(center,rotated));}}
  return candidate;
 }
 poremc::ChargeContribution translated_charge(size_t index,const std::vector<double>& candidate) const {
   const V displacement=sub(get(candidate,0),get(particles[index],0));
   return longrange->translated_contribution(particle_charge[index],{displacement.x,displacement.y,displacement.z});
 }
 bool step_reference_cbmc(size_t move){
  const auto& group=members[component()];
  const size_t n=group.size();auto*stats=acceleration_counters.data()+4*move;
  const bool resident=longrange&&longrange->resident();
  auto screen=[&](double ratio){
   ++stats[0];
   if(delayed_acceptance&&!accept(ratio))return false;
   ++stats[1];return true;
  };
  auto accept_exact=[&](double log_reference,double log_residual){
   const double ratio=delayed_acceptance ? log_residual : log_reference+log_residual;
   if(!accept(ratio)){
    if(resident)longrange->discard_resident_trial();
    return false;
   }
   ++stats[3];return true;
  };
  poremc::ChargeContribution prepared;double prepared_local=0,prepared_reference=0;
  if(move==0){
   if(mol->z==0)return false;
   if(particles.size()>=max_particles)throw std::runtime_error("FastGCMC max_particles reached; increase capacity, not a physical truncation");
   void*env=cached_environment();auto candidate=grow(env,nullptr,true);
   if(!candidate.valid)return false;
   double loga=std::log(mol->z)+std::log(Vbox)-std::log(double(n+1))+candidate.logw+std::log(probability[1]/probability[0]);
   if(!screen(loga))return false;
   wrap(candidate.x);++stats[2];
   double residual;
   if(resident){
    prepared_local=host_internal_energy(candidate.x);
    prepared_reference=reference_host_internal_energy(candidate.x);
    residual=prepared_local-prepared_reference+
             longrange->resident_trial(mol->A,candidate.x.data(),nullptr,mol->charge.data());
   }else residual=selected_residual(env,candidate.x,nullptr,&prepared,nullptr,&prepared_local,nullptr,&prepared_reference);
   if(!accept_exact(loga,-residual/T))return false;
   commit_insert(std::move(candidate.x),longrange&&!resident?&prepared:nullptr,&prepared_local,&prepared_reference);return true;
  }
  if(n==0)return false;
  size_t index=group[std::min(n-1,size_t(uniform()*n))];const auto&old=particles[index];
  void*env=cached_environment(index);
  const auto*old_charge=longrange?&particle_charge[index]:nullptr;
  if(move==1){
   auto retrace=grow(env,&old,true);if(!retrace.valid)throw std::runtime_error("invalid reference deletion retrace");
   double loga=mol->z==0?std::numeric_limits<double>::infinity():std::log(double(n))-std::log(mol->z)-std::log(Vbox)-retrace.logw+std::log(probability[0]/probability[1]);
   if(!screen(loga))return false;
   ++stats[2];
   const double residual=resident
       ? particle_host_internal[index]-particle_reference_local[index]-
         longrange->resident_trial(mol->A,nullptr,old.data(),mol->charge.data())
       : selected_residual(env,old,old_charge,&prepared,&particle_host_internal[index],
                           &prepared_local,&particle_reference_local[index],&prepared_reference);
   if(!std::isfinite(residual))throw std::runtime_error("invalid exact current molecule residual");
   if(!accept_exact(loga,residual/T))return false;
   commit_erase(index);return true;
  }
  std::vector<double> candidate;double old_ref=0,new_ref=0,loga=0;
  if(move==4){
   const auto start=regrowth_start();
   auto grown=grow(env,nullptr,true,start,&old);if(!grown.valid)return false;
   auto retrace=grow(env,&old,true,start);if(!retrace.valid)throw std::runtime_error("invalid reference regrowth retrace");
   old_ref=retrace.energy;new_ref=grown.energy;loga=grown.logw-retrace.logw;candidate=std::move(grown.x);
  }else{
   if(move==3&&mol->A==1)return false;
   candidate=local_proposal(old,move);
   old_ref=surrogate_molecule_energy(env,old);new_ref=surrogate_molecule_energy(env,candidate);
   if(!std::isfinite(old_ref))throw std::runtime_error("invalid reference current molecule energy");
   loga=-(new_ref-old_ref)/T;
  }
  if(!screen(loga))return false;
  wrap(candidate);
  stats[2]+=2;  // old/new molecular residuals; the old local terms are cached.
  double delta=0.;
  if(resident){
    prepared_local=host_internal_energy(candidate);
    prepared_reference=reference_host_internal_energy(candidate);
    delta=prepared_local-prepared_reference-(particle_host_internal[index]-particle_reference_local[index])+
          longrange->resident_trial(mol->A,candidate.data(),old.data(),mol->charge.data());
  }else if(longrange && longrange->config().method==0){
    prepared = move==2 ? translated_charge(index,candidate)
                       : longrange->selected_contribution(mol->A,candidate.data(),mol->charge.data());
    prepared_local=host_internal_energy(candidate);
    prepared_reference=reference_host_internal_energy(candidate);
    const double old_local=particle_host_internal[index]-particle_reference_local[index];
    delta=(prepared_local-prepared_reference)-old_local+longrange->replacement_energy(prepared,*old_charge);
  }else{
    const double old_residual=selected_residual(env,old,old_charge,&prepared,&particle_host_internal[index],&prepared_local,&particle_reference_local[index],&prepared_reference);
    const double new_residual=selected_residual(env,candidate,nullptr,&prepared,nullptr,&prepared_local,nullptr,&prepared_reference);
    if(!std::isfinite(old_residual))throw std::runtime_error("invalid exact current molecule residual");
    delta=new_residual-old_residual;
  }
  if(!accept_exact(loga,-delta/T))return false;
  commit_replace(index,std::move(candidate),longrange&&!resident?&prepared:nullptr,&prepared_local,&prepared_reference);return true;
 }
 void step(){double sel=uniform(),sum=0;size_t move=4;for(size_t m=0;m<5;++m){sum+=probability[m];if(sel<sum){move=m;break;}}++steps;++counters[2*move];
  if(molecules.size()>1){
   const auto choice=std::upper_bound(component_cdf.begin(),component_cdf.end(),uniform());
   select(std::min(molecules.size()-1,size_t(choice-component_cdf.begin())));
  }else select(0);
  bool accepted=false;
  const auto& group=members[component()];
  const size_t n=group.size();
  if(reference_cbmc){if(step_reference_cbmc(move))++counters[2*move+1];return;}
  if(move==0){if(mol->z==0)return;if(particles.size()>=max_particles)throw std::runtime_error("FastGCMC max_particles reached; increase capacity, not a physical truncation");void* env=cached_environment();auto candidate=grow(env);if(!candidate.valid)return;double ratio=std::log(mol->z)+std::log(Vbox)-std::log(static_cast<double>(n+1))+candidate.logw+std::log(probability[1]/probability[0]);accepted=accept(ratio);if(accepted){wrap(candidate.x);commit_insert(std::move(candidate.x));}}
  else {if(n==0)return;size_t index=group[std::min(n-1,static_cast<size_t>(uniform()*n))];const auto&old=particles[index];void* env=cached_environment(index);
   if(move==1){auto retrace=grow(env,&old);if(!retrace.valid)throw std::runtime_error("cannot retrace invalid current molecule");double ratio=mol->z==0?std::numeric_limits<double>::infinity():std::log(static_cast<double>(n))-std::log(mol->z)-std::log(Vbox)-retrace.logw+std::log(probability[0]/probability[1]);accepted=accept(ratio);if(accepted)commit_erase(index);}
   else if(move==4){const auto start=regrowth_start();auto candidate=grow(env,nullptr,false,start,&old);if(!candidate.valid)return;auto retrace=grow(env,&old,false,start);if(!retrace.valid)throw std::runtime_error("invalid old regrowth path");accepted=accept(candidate.logw-retrace.logw);if(accepted){wrap(candidate.x);commit_replace(index,std::move(candidate.x));}}
   else {
    if(move==3 && mol->A==1)return;
    auto candidate=local_proposal(old,move);
    // On CPU the accepted state already contains the old Fourier contribution.
    // Compute one spectrum and a single dot-product for the entire replacement.
    if(longrange && longrange->config().method==0 && longrange->config().backend==0){
      wrap(candidate);
      auto prepared = move==2 ? translated_charge(index,candidate)
                              : longrange->contribution(mol->A,candidate.data(),mol->charge.data());
      const double old_local=molecule_energy(env,old,false);
      const double new_local=molecule_energy(env,candidate,false);
      if(!std::isfinite(old_local))throw std::runtime_error("invalid current molecule energy");
      const double difference=new_local-old_local+longrange->replacement_energy(prepared,particle_charge[index]);
      accepted=accept(-difference/T);
      if(accepted)commit_replace(index,std::move(candidate),&prepared);
    }else{
      double old_u,new_u;
      const bool short_cuda=use_cuda_short_range(1);
      const bool reciprocal_cuda=longrange&&mol->has_partial_charge&&longrange->uses_cuda_batches();
      // Both poses use the same bath and excluded particle. Keep the original
      // short-range backend when dispatch would change at batch size two.
      if((short_cuda||reciprocal_cuda)&&short_cuda==use_cuda_short_range(2)){
       scratch_local_pair.assign(old.begin(),old.end());
       scratch_local_pair.insert(scratch_local_pair.end(),candidate.begin(),candidate.end());
       molecule_energies(env,scratch_local_pair.data(),2,scratch_values);
       old_u=scratch_values[0];new_u=scratch_values[1];
      }else{
       old_u=molecule_energy(env,old);new_u=molecule_energy(env,candidate);
      }
      if(!std::isfinite(old_u))throw std::runtime_error("invalid current molecule energy");
      accepted=accept(-(new_u-old_u)/T);
      if(accepted){wrap(candidate);commit_replace(index,std::move(candidate));}
    }
   }
  }if(accepted)++counters[2*move+1];
 }
 double total_energy(){
  double result=0.;
  for(size_t i=0;i<particles.size();++i){
   select(particle_type[i]);
   const auto& x=particles[i];
   if(guest_cells){
    double previous=0.;
    guest_cells->batch(1,mol->A,x.data(),mol->sigma.data(),mol->epsilon.data(),mol->charge.data(),
                       i,particles.size(),&previous);
    result+=host_internal_energy(x,false)+previous;
   }else{
    auto env=environment(std::numeric_limits<size_t>::max(),i);
    result+=molecule_energy(env.get(),x,false);
   }
  }
  if(longrange)result+=longrange->total_host_relative_energy();
  return result;
 }

};
}
extern "C" {
int poremc_fast_set_root(void*p,int nx,int ny,int nz,const double*prob){try{auto& s=*static_cast<Fast*>(p);for(auto& m:s.molecules)s.set_root(m,nx,ny,nz,prob);return 0;}catch(const std::exception&e){fast_error=e.what();return -1;}}
int poremc_fast_profile(void*p,uint64_t*out){if(!p||!out)return -1;auto&s=*static_cast<Fast*>(p);out[0]=s.guest_builds;out[1]=s.guest_reuses;out[2]=s.steps;return 0;}
size_t poremc_fast_config_size(){return sizeof(PoreMCFastConfig);}
const char* poremc_fast_last_error(){return fast_error.c_str();}
void* poremc_fast_create(const PoreMCFastConfig*c){try{fast_error.clear();if(!c)throw std::invalid_argument("null config");return new Fast(c,1,nullptr,0,1,true,false,nullptr,nullptr);}catch(const std::exception&e){fast_error=e.what();return nullptr;}}
void poremc_fast_destroy(void*p){delete static_cast<Fast*>(p);}
void* poremc_fast_create_components(const PoreMCFastConfig* configs,size_t count,const double* weights,
                                    int local,size_t pme_refresh,int background,int partial,
                                    const char* lj_prefix,const char* electrostatic_path){
 try{
  fast_error.clear();
  if(!configs||!count)throw std::invalid_argument("no component definitions");
  return new Fast(configs,count,weights,local,pme_refresh,background!=0,partial!=0,
                  lj_prefix,electrostatic_path);
 }catch(const std::exception& e){fast_error=e.what();return nullptr;}
}
int poremc_fast_add_component(void* p,size_t species,const double* coordinates,int raw){
 try{static_cast<Fast*>(p)->add_particle(species,coordinates,raw!=0);return 0;}
 catch(const std::exception& e){fast_error=e.what();return -1;}
}
int poremc_fast_add(void* p,const double* x){return poremc_fast_add_component(p,0,x,0);}
int poremc_fast_add_raw(void* p,const double* x){return poremc_fast_add_component(p,0,x,1);}
size_t poremc_fast_count(void* p){return static_cast<Fast*>(p)->particles.size();}
int poremc_fast_counts(void* p,uint64_t* counts){
 const auto& s=*static_cast<Fast*>(p);
 for(size_t i=0;i<s.members.size();++i)counts[i]=s.members[i].size();
 return 0;
}
int poremc_fast_set_activities(void* p,const double* activities){
 try{
  auto& s=*static_cast<Fast*>(p);
  for(size_t i=0;i<s.molecules.size();++i)
   if(!std::isfinite(activities[i])||activities[i]<0.||(!activities[i]&&!s.members[i].empty()))
    throw std::invalid_argument("invalid activity or occupied zero-activity component");
  for(size_t i=0;i<s.molecules.size();++i)s.molecules[i].z=activities[i];
  return 0;
 }catch(const std::exception& e){fast_error=e.what();return -1;}
}
uint64_t poremc_fast_step_count(void* p){return static_cast<Fast*>(p)->steps;}
int poremc_fast_export_particles(void* p,uint64_t* species,double* xyz){
 const auto& s=*static_cast<Fast*>(p);
 for(size_t i=0;i<s.particles.size();++i){
  if(species)species[i]=s.particle_type[i];
  xyz=std::copy(s.particles[i].begin(),s.particles[i].end(),xyz);
 }
 return 0;
}
int poremc_fast_positions(void* p,double* out){
 if(static_cast<Fast*>(p)->molecules.size()!=1){fast_error="positions requires a single component";return -1;}
 return poremc_fast_export_particles(p,nullptr,out);
}
uint64_t poremc_fast_guest_updates(void* p){return static_cast<Fast*>(p)->guest_updates;}
int poremc_fast_counters(void* p,uint64_t* out){auto& s=*static_cast<Fast*>(p);std::copy(s.counters.begin(),s.counters.end(),out);return 0;}
int poremc_fast_run_components(void* p,uint64_t burn,uint64_t steps,uint64_t every,uint64_t* counts,
                               double* energies,int record,double* timing){
 try{
  if(every==0)throw std::invalid_argument("sample_every is zero");
  auto& s=*static_cast<Fast*>(p);
  const auto start=std::chrono::steady_clock::now();
  for(uint64_t i=0;i<burn;++i)s.step();
  const auto production=std::chrono::steady_clock::now();
  uint64_t out=0;
  for(uint64_t i=0;i<steps;++i){
   s.step();
   if((i+1)%every==0){
    poremc_fast_counts(p,counts+out*s.members.size());
    if(record)energies[out]=s.total_energy();
    ++out;
   }
  }
  const auto end=std::chrono::steady_clock::now();
  timing[0]=std::chrono::duration<double>(end-production).count();
  timing[1]=std::chrono::duration<double>(production-start).count();
  return 0;
 }catch(const std::exception& e){fast_error=e.what();return -1;}
}
int poremc_fast_run(void* p,uint64_t burn,uint64_t steps,uint64_t every,
                   uint64_t* counts,double* energies,int record,double* timing){
 if(static_cast<Fast*>(p)->molecules.size()!=1){fast_error="use run_components for mixtures";return -1;}
 return poremc_fast_run_components(p,burn,steps,every,counts,energies,record,timing);
}
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
    <<",\"gpu_batches\":"<<st.gpu_batches<<",\"gpu_selected_structures\":"<<st.gpu_selected_structures<<",\"device\":"<<std::quoted(s.longrange->device_name())<<",\"modes\":"<<st.modes<<",\"full_modes\":"<<st.full_modes
    <<",\"half_spectrum\":"<<(s.longrange->config().half_spectrum?"true":"false")
    <<",\"pme_refresh\":"<<s.longrange->config().pme_refresh<<",\"pme_pending_nodes\":"<<st.pme_pending_nodes
    <<",\"resident_trials\":"<<st.resident_trials<<",\"resident_commits\":"<<st.resident_commits
    <<",\"k_cutoff\":"<<s.longrange->config().k_cutoff<<",\"host_grid_builds\":"<<st.host_grid_builds
    <<",\"host_field_evaluations\":"<<st.host_field_evaluations<<",\"mesh_points\":"<<st.mesh_points<<"}";
  }
  std::string info=out.str();if(buffer&&capacity>=info.size()+1)std::memcpy(buffer,info.c_str(),info.size()+1);return info.size()+1;
 }catch(const std::exception&e){fast_error=e.what();return 0;}
}
int poremc_fast_short_range_enable(void* p,const char* library,size_t min_batch){
 try{
  if(!p||!min_batch)throw std::invalid_argument("CUDA short-range requires a positive minimum batch");
  auto& sim=*static_cast<Fast*>(p);
  if(sim.short_range_cuda||sim.steps||!sim.particles.empty())
   throw std::invalid_argument("select CUDA short-range before starting a simulation");
  auto* context=poremc_short_cuda_create(sim.host,library);
  if(!context)throw std::runtime_error(poremc_last_error());
  sim.short_range_cuda.reset(context);
  sim.short_range_min_batch=min_batch;
  return 0;
 }catch(const std::exception& error){fast_error=error.what();return -1;}
}
size_t poremc_fast_short_range_info(void* p,char* buffer,size_t capacity){
 try{
  if(!p)throw std::invalid_argument("closed FastGCMC instance");
  const auto& sim=*static_cast<Fast*>(p);
  std::ostringstream out;
  if(sim.short_range_cuda){
   out<<"{\"backend\":\"cuda\",\"device\":"<<std::quoted(poremc_short_cuda_device(sim.short_range_cuda.get()))
      <<",\"min_batch\":"<<sim.short_range_min_batch
      <<",\"framework_batches\":"<<sim.cuda_framework_batches
      <<",\"guest_batches\":"<<sim.cuda_guest_batches
      <<",\"guest_uploads\":"<<sim.cuda_guest_uploads<<'}';
  }else out<<"{\"backend\":\"cpu\"}";
  auto info=out.str();
  if(buffer&&capacity>=info.size()+1)std::memcpy(buffer,info.c_str(),info.size()+1);
  return info.size()+1;
 }catch(const std::exception& error){fast_error=error.what();return 0;}
}
int poremc_fast_probe_sites(void* p,size_t species,size_t exclude,size_t i,const double* prefix,
                            size_t batches,const double* trials,int reference,double* out){
 try{
  auto& s=*static_cast<Fast*>(p);
  s.select(species);
  if(reference&&!s.mol->host_grid)throw std::invalid_argument("surrogate probes require a reference CBMC mode");
  if(i>=s.mol->A)throw std::invalid_argument("prefix length must be below atom count");
  if(exclude!=std::numeric_limits<size_t>::max()&&exclude>=s.particles.size())
   throw std::invalid_argument("excluded molecule index is out of range");
  if(!batches)return 0;
  std::vector<double> pre;
  if(i)pre.assign(prefix,prefix+3*i);
  std::vector<double> xyz(trials,trials+3*batches),energies;
  s.site_energies(s.cached_environment(exclude),pre,i,xyz,energies,nullptr,reference!=0);
  std::copy(energies.begin(),energies.end(),out);
  return 0;
 }catch(const std::exception& e){fast_error=e.what();return -1;}
}
int poremc_fast_probe_molecules(void* p,size_t species,size_t exclude,size_t batches,
                                const double* trials,int reference,double* out){
 try{
  auto& s=*static_cast<Fast*>(p);
  s.select(species);
  if(reference&&!s.mol->host_grid)throw std::invalid_argument("surrogate probes require a reference CBMC mode");
  if(exclude!=std::numeric_limits<size_t>::max()&&exclude>=s.particles.size())
   throw std::invalid_argument("excluded molecule index is out of range");
  if(!batches)return 0;
  std::vector<double> energies;
  void* env=s.cached_environment(exclude);
  if(reference)s.surrogate_molecule_energies(env,trials,batches,energies);
  else s.molecule_energies(env,trials,batches,energies);
  std::copy(energies.begin(),energies.end(),out);
  return 0;
 }catch(const std::exception& e){fast_error=e.what();return -1;}
}
int poremc_fast_site_energies(void* p,size_t exclude,size_t i,const double* prefix,size_t batches,const double* trials,double* out){
 return poremc_fast_probe_sites(p,0,exclude,i,prefix,batches,trials,0,out);
}
int poremc_fast_molecule_energies(void* p,size_t exclude,size_t batches,const double* trials,double* out){
 return poremc_fast_probe_molecules(p,0,exclude,batches,trials,0,out);
}
int poremc_fast_surrogate_site_energies(void* p,size_t exclude,size_t i,const double* prefix,size_t batches,const double* trials,double* out){
 return poremc_fast_probe_sites(p,0,exclude,i,prefix,batches,trials,1,out);
}
int poremc_fast_surrogate_molecule_energies(void* p,size_t exclude,size_t batches,const double* trials,double* out){
 return poremc_fast_probe_molecules(p,0,exclude,batches,trials,1,out);
}
int poremc_fast_acceleration_counters(void*p,uint64_t*out){if(!p||!out)return -1;auto&s=*static_cast<Fast*>(p);std::copy(s.acceleration_counters.begin(),s.acceleration_counters.end(),out);return 0;}
int poremc_fast_restore_acceleration_counters(void*p,const uint64_t*in){if(!p||!in)return -1;auto&s=*static_cast<Fast*>(p);std::copy(in,in+20,s.acceleration_counters.begin());return 0;}
size_t poremc_fast_acceleration_info(void*p,char*buffer,size_t capacity){
 try{auto&s=*static_cast<Fast*>(p);std::ostringstream out;
  const char* selected="not_used";
  if(s.reference_cbmc){
   selected="cpu";
   if(s.longrange&&s.longrange->resident())selected="cuda-resident";
   else if(s.longrange&&std::string(s.longrange->backend_name())=="cuda-tensor")selected="cuda-tensor-sf/cpu-residual";
  }
  const bool single=s.reference_cbmc&&!s.delayed_acceptance;
  const char* evaluated_key=single?"reference_proposals":"stage1_evaluated";
  const char* passed_key=single?"proposals_reaching_exact":"stage1_passed";
  const char* accepted_key=single?"accepted":"stage2_accepted";
  out<<"{\"sampling_mode\":\""<<(single?"reference-mh":s.delayed_acceptance?"delayed":"exact")<<"\",\"selected_correction_backend\":\""<<selected<<"\",\"grid_root_bias\":"<<s.root_bias<<",\"stages\":{";
  const char*names[5]={"insert","delete","translate","rotate","regrow"};
  uint64_t evaluated=0,passed=0,exact=0,accepted=0;
  for(size_t m=0;m<5;++m){auto*c=s.acceleration_counters.data()+4*m;if(m)out<<',';
   out<<'\"'<<names[m]<<"\":{\""<<evaluated_key<<"\":"<<c[0]<<",\""<<passed_key<<"\":"<<c[1]<<",\"exact_evaluations\":"<<c[2]<<",\""<<accepted_key<<"\":"<<c[3]<<'}';
   evaluated+=c[0];passed+=c[1];exact+=c[2];accepted+=c[3];
  }
  out<<"},\""<<evaluated_key<<"\":"<<evaluated<<",\""<<passed_key<<"\":"<<passed<<",\"exact_evaluations\":"<<exact<<",\""<<accepted_key<<"\":"<<accepted;
  if(s.reference_cbmc){
   poremc::HostGridStats st;
   for(const auto& m:s.molecules){
    const auto g=m.host_grid->stats();
    st.builds+=g.builds;
    st.cache_loads+=g.cache_loads;
    st.cache_writes+=g.cache_writes;
    st.lookups+=g.lookups;
    st.cubic_lookups+=g.cubic_lookups;
    st.linear_lookups+=g.linear_lookups;
    st.boltzmann_lookups+=g.boltzmann_lookups;
    st.exact_fallbacks+=g.exact_fallbacks;
    st.lj_fields+=g.lj_fields;
    st.site_types+=g.site_types;
    st.bytes+=g.bytes;
    st.steep_cells+=g.steep_cells;
    st.raw_field_builds+=g.raw_field_builds;
    st.raw_field_loads+=g.raw_field_loads;
    st.mesh_points=g.mesh_points;
   }
   const auto& grid=*s.molecules.front().host_grid;
   out<<",\"host_grid\":{\"builds\":"<<st.builds<<",\"cache_loads\":"<<st.cache_loads<<",\"cache_writes\":"<<st.cache_writes
   <<",\"raw_field_builds\":"<<st.raw_field_builds<<",\"raw_field_loads\":"<<st.raw_field_loads
   <<",\"lookups\":"<<st.lookups<<",\"cubic_lookups\":"<<st.cubic_lookups<<",\"linear_lookups\":"<<st.linear_lookups<<",\"boltzmann_lookups\":"<<st.boltzmann_lookups<<",\"exact_fallbacks\":"<<st.exact_fallbacks
   <<",\"steep_interpolation\":\""<<(grid.config().steep_interpolation==1?"boltzmann":"linear")<<"\",\"reference_cap_K\":"<<grid.config().surrogate_cap
   <<",\"mesh_points\":"<<st.mesh_points<<",\"lj_fields\":"<<st.lj_fields<<",\"site_types\":"<<st.site_types<<",\"bytes\":"<<st.bytes<<",\"steep_cells\":"<<st.steep_cells<<"}";}
  out<<'}';auto info=out.str();if(buffer&&capacity>=info.size()+1)std::memcpy(buffer,info.c_str(),info.size()+1);return info.size()+1;
 }catch(const std::exception&e){fast_error=e.what();return 0;}
}
}
