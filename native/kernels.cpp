// PoreMC: independently implemented pair-energy kernels. MIT license.
// Cell vectors are rows. Energies U/k_B are in K; lengths are in Angstrom.
#include "guest_cells.h"
#include "short_range_cuda.h"
#include <algorithm>
#include <atomic>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <limits>
#include <numeric>
#include <map>
#include <mutex>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(__unix__) || defined(__APPLE__)
#include <dlfcn.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

namespace {
constexpr double COULOMB_K = 167100.94689828736;
constexpr double INV_SQRT_PI = 0.56418958354775628695;
thread_local std::string last_error;
struct PairTable {std::vector<double> s2,e4,shift,q;};
struct Environment {
  std::array<double,9> cell{}, inv{};
  std::array<double,3> heights{}, lengths{};
  std::array<int,3> bins{};
  std::vector<double> sx, sy, sz, sigma, sqrt_epsilon, charge;
  std::vector<std::size_t> starts, adj_start;
  std::vector<std::uint32_t> original_id, adj;
  std::map<std::array<double,3>,std::shared_ptr<PairTable>> tables;
  std::mutex mutex;
  std::uint64_t hits=0,misses=0,table_bytes=0;
  std::atomic<std::uint64_t> calls{0};
  bool expanded=false;
  double cutoff, cutoff2, inv_cutoff2, alpha, dsf_value, dsf_slope;
  int coulomb, shifted, algorithm, threads;
  bool diagonal;
  std::size_t m;

  Environment(std::size_t count,const double* pos,const double* s,const double* e,
      const double* q,const double* h,double rc,int cm,double a,int sh,int alg,int nt)
      : cutoff(rc),cutoff2(rc*rc),inv_cutoff2(1/(rc*rc)),alpha(a),
        coulomb(cm),shifted(sh),algorithm(alg),threads(nt),m(count) {
    std::copy(h,h+9,cell.begin());
    const double det=h[0]*(h[4]*h[8]-h[5]*h[7])-h[1]*(h[3]*h[8]-h[5]*h[6])+h[2]*(h[3]*h[7]-h[4]*h[6]);
    if (!(det>0) || !std::isfinite(det)) throw std::invalid_argument("cell must have finite positive determinant");
    inv={(h[4]*h[8]-h[5]*h[7])/det,(h[2]*h[7]-h[1]*h[8])/det,(h[1]*h[5]-h[2]*h[4])/det,
         (h[5]*h[6]-h[3]*h[8])/det,(h[0]*h[8]-h[2]*h[6])/det,(h[2]*h[3]-h[0]*h[5])/det,
         (h[3]*h[7]-h[4]*h[6])/det,(h[1]*h[6]-h[0]*h[7])/det,(h[0]*h[4]-h[1]*h[3])/det};
    if (!(rc>0) || !std::isfinite(rc) || !(a>=0) || !std::isfinite(a)) throw std::invalid_argument("invalid cutoff or alpha");
    for(int d=0;d<3;++d) {
      heights[d]=1/std::sqrt(inv[d]*inv[d]+inv[3+d]*inv[3+d]+inv[6+d]*inv[6+d]);
      if (!(2*rc<heights[d])) throw std::invalid_argument("cutoff must be strictly less than half every cell face height");
      bins[d]=std::max(1,std::min(128,static_cast<int>(std::min(128.0,std::floor((alg==2?2.0:1.0)*heights[d]/rc)))));
      lengths[d]=cell[d*3+d];
    }
    diagonal=(h[1]==0 && h[2]==0 && h[3]==0 && h[5]==0 && h[6]==0 && h[7]==0);
    if(cm<0 || cm>2 || alg<0 || alg>2 || nt<1) throw std::invalid_argument("invalid mode or thread count");
    dsf_value=std::erfc(a*rc)/rc;
    dsf_slope=std::erfc(a*rc)/(rc*rc)+2*a*INV_SQRT_PI*std::exp(-a*a*rc*rc)/rc;
    std::vector<std::array<double,3>> fractional(m);
    std::vector<std::size_t> keys(m),order(m);
    for(std::size_t j=0;j<m;++j) {
      for(int d=0;d<3;++d) {
        double v=pos[j*3]*inv[d]+pos[j*3+1]*inv[3+d]+pos[j*3+2]*inv[6+d];
        fractional[j][d]=v-std::floor(v);
      }
      int ix=std::min(bins[0]-1,static_cast<int>(fractional[j][0]*bins[0]));
      int iy=std::min(bins[1]-1,static_cast<int>(fractional[j][1]*bins[1]));
      int iz=std::min(bins[2]-1,static_cast<int>(fractional[j][2]*bins[2]));
      keys[j]=(static_cast<std::size_t>(ix)*bins[1]+iy)*bins[2]+iz;
      if (!(s[j]>=0) || !(e[j]>=0) || !std::isfinite(s[j]) || !std::isfinite(e[j]) || !std::isfinite(q[j])) throw std::invalid_argument("invalid atom parameter");
    }
    std::iota(order.begin(),order.end(),0);
    if(algorithm!=0 && m>=64) std::stable_sort(order.begin(),order.end(),[&](auto i,auto j){return keys[i]<keys[j];});
    original_id.resize(m);sx.resize(m);sy.resize(m);sz.resize(m);sigma.resize(m);sqrt_epsilon.resize(m);charge.resize(m);
    const std::size_t ncell=static_cast<std::size_t>(bins[0])*bins[1]*bins[2];
    if(algorithm!=0 && m>=64) starts.assign(ncell+1,0);
    for(std::size_t k=0;k<m;++k) {
      auto j=order[k];original_id[k]=static_cast<std::uint32_t>(j);sx[k]=fractional[j][0];sy[k]=fractional[j][1];sz[k]=fractional[j][2];
      sigma[k]=s[j];sqrt_epsilon[k]=std::sqrt(e[j]);charge[k]=q[j];
      if(algorithm!=0 && m>=64) ++starts[keys[j]+1];
    }
    if(algorithm!=0 && m>=64) std::partial_sum(starts.begin(),starts.end(),starts.begin());
    if(algorithm==2 && m>=64) {
      std::vector<int> offsets[3];
      for(int d=0;d<3;++d) {
        int rad=static_cast<int>(std::ceil(cutoff*bins[d]/heights[d]));
        for(int k=0;k<bins[d];++k) {int v=k<=bins[d]/2?k:k-bins[d];if(std::abs(v)<=rad)offsets[d].push_back(v);}
      }
      std::vector<std::array<int,3>> stencil;
      for(int x:offsets[0])for(int y:offsets[1])for(int z:offsets[2]) {
        if(diagonal) {
          double dx=std::max(0,std::abs(x)-1)*lengths[0]/bins[0],dy=std::max(0,std::abs(y)-1)*lengths[1]/bins[1],dz=std::max(0,std::abs(z)-1)*lengths[2]/bins[2];
          if(dx*dx+dy*dy+dz*dz>=cutoff2)continue;
        }
        stencil.push_back({x,y,z});
      }
      if(m<=UINT32_MAX && stencil.size()*m*4ull<=64ull*1024*1024) {
        adj_start.resize(ncell+1);adj.reserve(stencil.size()*m);
        for(int x=0;x<bins[0];++x)for(int y=0;y<bins[1];++y)for(int z=0;z<bins[2];++z) {
          size_t key=(static_cast<size_t>(x)*bins[1]+y)*bins[2]+z;adj_start[key]=adj.size();
          for(auto d:stencil) {
            int nx=(x+d[0]+bins[0])%bins[0],ny=(y+d[1]+bins[1])%bins[1],nz=(z+d[2]+bins[2])%bins[2];
            size_t k=(static_cast<size_t>(nx)*bins[1]+ny)*bins[2]+nz;
            for(size_t j=starts[k];j<starts[k+1];++j)adj.push_back(static_cast<std::uint32_t>(j));
          }
        }
        adj_start[ncell]=adj.size();expanded=true;
      }
    }
  }
  std::shared_ptr<PairTable> bind(double s,double e,double q) {
    if(algorithm!=2 || m==0)return {};
    std::array<double,3> key{s,e,q};std::lock_guard<std::mutex> lock(mutex);
    auto it=tables.find(key);if(it!=tables.end()){++hits;return it->second;}
    if(table_bytes+32*m>64ull*1024*1024)return {};
    auto t=std::make_shared<PairTable>();t->s2.resize(m);t->e4.resize(m);t->shift.resize(m);t->q.resize(m);
    double se=std::sqrt(e);
    for(size_t j=0;j<m;++j) {
      double a=.5*(s+sigma[j]);t->s2[j]=a*a;t->e4[j]=4*se*sqrt_epsilon[j];
      double c=a*a*inv_cutoff2,c3=c*c*c;
      t->shift[j]=shifted?t->e4[j]*c3*(c3-1):0.;t->q[j]=COULOMB_K*q*charge[j];
    }
    tables.emplace(key,t);table_bytes+=32*m;++misses;return t;
  }

  inline std::array<double,3> fractional(const double* xyz) const {
    std::array<double,3> f{};
    for(int d=0;d<3;++d) {
      f[d]=xyz[0]*inv[d]+xyz[1]*inv[3+d]+xyz[2]*inv[6+d];
      f[d]-=std::floor(f[d]);
    }
    return f;
  }

  inline double distance_squared(double fx,double fy,double fz) const {
    fx-=std::nearbyint(fx);fy-=std::nearbyint(fy);fz-=std::nearbyint(fz);
    double dx,dy,dz;
    if(diagonal) {dx=fx*lengths[0];dy=fy*lengths[1];dz=fz*lengths[2];}
    else {
      dx=fx*cell[0]+fy*cell[3]+fz*cell[6];
      dy=fx*cell[1]+fy*cell[4]+fz*cell[7];
      dz=fx*cell[2]+fy*cell[5]+fz*cell[8];
    }
    return dx*dx+dy*dy+dz*dz;
  }

  inline double pair_energy(double r2,double s2,double e4,double qq,
                            const double* cached_shift=nullptr) const {
    double v=0.;
    if(e4>0 && s2>0) {
      if(r2==0)return std::numeric_limits<double>::infinity();
      const double t=s2/r2,t3=t*t*t;
      v=e4*t3*(t3-1);
      if(cached_shift)v-=*cached_shift;
      else if(shifted) {
        const double c=s2*inv_cutoff2,c3=c*c*c;
        v-=e4*c3*(c3-1);
      }
      if(shifted==2) {
        const double c=s2*inv_cutoff2,c3=c*c*c;
        v-=6*e4*c3*(1-2*c3)*(std::sqrt(r2)-cutoff)/cutoff;
      }
    }
    if(coulomb && qq!=0. && !std::isinf(v)) {
      if(r2==0)return std::copysign(std::numeric_limits<double>::infinity(),qq);
      const double r=std::sqrt(r2),radial=std::erfc(alpha*r)/r;
      v+=qq*(coulomb==1?radial-dsf_value+dsf_slope*(r-cutoff):radial);
    }
    return v;
  }

  inline double range_energy(double x,double y,double z,double si,double sqei,double qi,
                             std::size_t begin,std::size_t end,size_t exlo=0,size_t exhi=0,const PairTable* table=nullptr,const std::uint32_t* ids=nullptr) const {
    double total=0;
    // omp simd uses full-precision arithmetic (no -ffast-math). Different legal
    // reduction orders may differ by round-off; accepted moves use double precision.
    #pragma omp simd reduction(+:total)
    for(std::size_t k=begin;k<end;++k) {
      const std::size_t j=ids?ids[k]:k;
      if(exhi>exlo && original_id[j]>=exlo && original_id[j]<exhi)continue;
      const double r2=distance_squared(x-sx[j],y-sy[j],z-sz[j]);
      if(r2<cutoff2) {
        const double sij=table?0.0:0.5*(si+sigma[j]);
        const double s2=table?table->s2[j]:sij*sij,e4=table?table->e4[j]:4*sqei*sqrt_epsilon[j];
        const double qq=table?table->q[j]:COULOMB_K*qi*charge[j];
        total+=pair_energy(r2,s2,e4,qq,table?&table->shift[j]:nullptr);
      }
    }
    return total;
  }

  inline double atom_energy(const double* xyz,double si,double ei,double qi,size_t exlo=0,size_t exhi=0,const PairTable* table=nullptr) const {
    if(m==0) return 0.0;
    const auto f=fractional(xyz);
    const double sqei=std::sqrt(ei);
    if(algorithm==0 || m<64) return range_energy(f[0],f[1],f[2],si,sqei,qi,0,m,exlo,exhi,table);
    if(expanded) {
      int ix=std::min(bins[0]-1,int(f[0]*bins[0])),iy=std::min(bins[1]-1,int(f[1]*bins[1])),iz=std::min(bins[2]-1,int(f[2]*bins[2]));
      size_t key=(size_t(ix)*bins[1]+iy)*bins[2]+iz;
      return range_energy(f[0],f[1],f[2],si,sqei,qi,adj_start[key],adj_start[key+1],exlo,exhi,table,adj.data());
    }
    int center[3],nb[3][7],nn[3];
    for(int d=0;d<3;++d) {
      center[d]=std::min(bins[d]-1,static_cast<int>(f[d]*bins[d]));nn[d]=0;
      for(int delta=-(algorithm==2?2:1);delta<=(algorithm==2?2:1);++delta) {
        const int t=(center[d]+delta+bins[d])%bins[d];bool seen=false;
        for(int k=0;k<nn[d];++k) if(nb[d][k]==t) seen=true;
        if(!seen) nb[d][nn[d]++]=t;
      }
    }
    double total=0;
    for(int a=0;a<nn[0];++a)for(int b=0;b<nn[1];++b)for(int c=0;c<nn[2];++c) {
      const std::size_t key=(static_cast<std::size_t>(nb[0][a])*bins[1]+nb[1][b])*bins[2]+nb[2][c];
      total+=range_energy(f[0],f[1],f[2],si,sqei,qi,starts[key],starts[key+1],exlo,exhi,table);
    }
    return total;
  }
};
}

namespace poremc {
struct GuestCells::Impl {
  struct Site {
    std::size_t owner;
    std::array<double,3> position;
    double sigma, sqrt_epsilon, charge;
  };
  Environment geometry;
  std::array<int,3> bins;
  std::vector<std::vector<Site>> cells;
  std::array<std::vector<std::array<int,3>>,3> neighbors;
  std::array<std::vector<int>,3> widths;

  Impl(const double* h,double rc,int coulomb,double alpha,int shifted)
      : geometry(0,nullptr,nullptr,nullptr,nullptr,h,rc,coulomb,alpha,shifted,1,1),
        bins(geometry.bins) {
    for(auto& n:bins)n=std::min(n,32);
    cells.resize(std::size_t(bins[0])*bins[1]*bins[2]);
    for(int d=0;d<3;++d) {
      const int n=bins[d];
      neighbors[d].resize(n);
      widths[d].resize(n,n==2?2:3);
      for(int i=0;i<n;++i)neighbors[d][i]={(i+n-1)%n,i,(i+1)%n};
    }
  }

  std::array<int,3> center(const std::array<double,3>& f) const {
    return {std::min(bins[0]-1,int(f[0]*bins[0])),
            std::min(bins[1]-1,int(f[1]*bins[1])),
            std::min(bins[2]-1,int(f[2]*bins[2]))};
  }
  std::size_t key(int i,int j,int k) const {
    return (std::size_t(i)*bins[1]+j)*bins[2]+k;
  }
  std::vector<Site>& box(const std::array<double,3>& f) {
    const auto c=center(f);
    return cells[key(c[0],c[1],c[2])];
  }
  double energy(const double* xyz,double sigma,double sqrt_epsilon,double q,
                std::size_t lo,std::size_t hi) const {
    const auto f=geometry.fractional(xyz);
    const auto c=center(f);
    double total=0.;
    for(int i=0;i<widths[0][c[0]];++i)
      for(int j=0;j<widths[1][c[1]];++j)
        for(int k=0;k<widths[2][c[2]];++k) {
          const auto& box=cells[key(neighbors[0][c[0]][i],neighbors[1][c[1]][j],neighbors[2][c[2]][k])];
          double partial=0.;
          for(const auto& site:box) {
            if(site.owner>=lo&&site.owner<hi)continue;
            const double r2=geometry.distance_squared(f[0]-site.position[0],f[1]-site.position[1],f[2]-site.position[2]);
            if(r2>=geometry.cutoff2)continue;
            const double s=.5*(sigma+site.sigma);
            partial+=geometry.pair_energy(r2,s*s,4*sqrt_epsilon*site.sqrt_epsilon,COULOMB_K*q*site.charge);
          }
          total+=partial;
        }
    return total;
  }
};

GuestCells::GuestCells(const double* h,double rc,int coulomb,double alpha,int shifted)
    :impl_(new Impl(h,rc,coulomb,alpha,shifted)) {}
GuestCells::~GuestCells()=default;
void GuestCells::reserve(std::size_t atoms,const double* xyz) {
  for(std::size_t i=0;i<atoms;++i) {
    auto& box=impl_->box(impl_->geometry.fractional(xyz+3*i));
    if(box.capacity()-box.size()<atoms)box.reserve(std::max(box.size()+atoms,box.capacity()*2));
  }
}
void GuestCells::insert(std::size_t owner,std::size_t atoms,const double* xyz,
                        const double* sigma,const double* epsilon,const double* q) {
  for(std::size_t i=0;i<atoms;++i) {
    const auto f=impl_->geometry.fractional(xyz+3*i);
    auto& box=impl_->box(f);
    const auto pos=std::upper_bound(box.begin(),box.end(),owner,
        [](std::size_t id,const Impl::Site& site){return id<site.owner;});
    box.insert(pos,{owner,f,sigma[i],std::sqrt(epsilon[i]),q[i]});
  }
}
void GuestCells::erase(std::size_t owner,std::size_t atoms,const double* xyz) {
  for(std::size_t i=0;i<atoms;++i) {
    auto& box=impl_->box(impl_->geometry.fractional(xyz+3*i));
    auto first=std::lower_bound(box.begin(),box.end(),owner,
        [](const Impl::Site& site,std::size_t id){return site.owner<id;});
    auto end=first;
    while(end!=box.end()&&end->owner==owner)++end;
    box.erase(first,end);
  }
}
void GuestCells::batch(std::size_t batches,std::size_t atoms,const double* xyz,
                      const double* sigma,const double* epsilon,const double* q,
                      std::size_t lo,std::size_t hi,double* output) const {
  std::fill(output,output+batches,0.);
  for(std::size_t a=0;a<atoms;++a) {
    const double se=std::sqrt(epsilon[a]);
    for(std::size_t b=0;b<batches;++b)
      output[b]+=impl_->energy(xyz+3*(b*atoms+a),sigma[a],se,q[a],lo,hi);
  }
}
}

extern "C" {
const char* poremc_last_error() {return last_error.c_str();}
const char* poremc_kernel_version() {return "OpenGCMC native C++17 FP64";}
int poremc_kernel_abi() {return 2;}
int poremc_has_openmp() {
#ifdef _OPENMP
 return 1;
#else
 return 0;
#endif
}
void* poremc_environment_create(std::size_t count,const double* pos,const double* sigma,const double* epsilon,
 const double* charge,const double* cell,double cutoff,int coulomb,double alpha,int shifted,int algorithm,int threads) {
  try {last_error.clear();return new Environment(count,pos,sigma,epsilon,charge,cell,cutoff,coulomb,alpha,shifted,algorithm,threads);}
  catch(const std::exception& e){last_error=e.what();return nullptr;}
}
void poremc_environment_destroy(void* p) {delete static_cast<Environment*>(p);}
int poremc_environment_energy_excluding(void* p,size_t batches,size_t atoms,const double* xyz,const double* sigma,
 const double* epsilon,const double* charge,size_t exlo,size_t exhi,double* out) {
 try {
  if(!p)throw std::invalid_argument("closed environment");auto& env=*static_cast<Environment*>(p);
  env.calls.fetch_add(1,std::memory_order_relaxed);
  if(env.m==0 || atoms==0){std::fill(out,out+batches,0.0);return 0;}
  std::array<std::shared_ptr<PairTable>,16> local_tables;
  std::vector<std::shared_ptr<PairTable>> large_tables;
  if(atoms>16)large_tables.resize(atoms);
  auto* tables=atoms>16?large_tables.data():local_tables.data();
  if(env.algorithm==2)for(size_t a=0;a<atoms;++a)tables[a]=env.bind(sigma[a],epsilon[a],charge[a]);
  #pragma omp parallel for schedule(static) num_threads(env.threads) if(batches>=32 && env.threads>1)
  for(std::ptrdiff_t b=0;b<static_cast<std::ptrdiff_t>(batches);++b) {
   double total=0;for(size_t a=0;a<atoms;++a)total+=env.atom_energy(xyz+(size_t(b)*atoms+a)*3,sigma[a],epsilon[a],charge[a],exlo,exhi,tables[a].get());out[b]=total;
  }return 0;
 }catch(const std::exception& e){last_error=e.what();return -1;}
}
int poremc_environment_energy(void* p,size_t batches,size_t atoms,const double* xyz,const double* sigma,
 const double* epsilon,const double* charge,double* out) {
 return poremc_environment_energy_excluding(p,batches,atoms,xyz,sigma,epsilon,charge,0,0,out);
}
int poremc_environment_profile(void* p,std::uint64_t* out) {
 if(!p||!out)return -1;auto& e=*static_cast<Environment*>(p);std::lock_guard<std::mutex> lock(e.mutex);
 out[0]=e.calls.load(std::memory_order_relaxed);out[1]=e.hits;out[2]=e.misses;out[3]=e.table_bytes;
 out[4]=e.adj.size()*4+e.adj_start.size()*sizeof(size_t);out[5]=e.adj.size();out[6]=e.adj_start.empty()?0:e.adj_start.size()-1;out[7]=e.expanded;return 0;
}
}
extern "C" int poremc_environment_export_plan(void* p,int* bins,std::uint32_t* order,std::uint64_t* offsets,std::uint32_t* indices) {
 if(!p)return -1;auto&e=*static_cast<Environment*>(p);
 std::copy(e.bins.begin(),e.bins.end(),bins);
 std::copy(e.original_id.begin(),e.original_id.end(),order);
 if(e.expanded){std::copy(e.adj_start.begin(),e.adj_start.end(),offsets);std::copy(e.adj.begin(),e.adj.end(),indices);}return 0;
}

// Bound query owns parameter rows, but borrows the immutable environment.
// Its lifetime must end before the environment is destroyed.
namespace {
struct BoundQuery {
 Environment* env;size_t atoms;
 std::vector<double> sigma,epsilon,charge;
 std::vector<std::shared_ptr<PairTable>> tables;
 BoundQuery(Environment* e,size_t n,const double*s,const double*ep,const double*q)
  :env(e),atoms(n),sigma(s,s+n),epsilon(ep,ep+n),charge(q,q+n),tables(n){
  for(size_t i=0;i<n;++i)tables[i]=env->bind(s[i],ep[i],q[i]);
 }
};
}
extern "C" void* poremc_query_create(void*p,size_t n,const double*s,const double*e,const double*q){
 try{if(!p)throw std::invalid_argument("null environment");return new BoundQuery(static_cast<Environment*>(p),n,s,e,q);}
 catch(const std::exception&e){last_error=e.what();return nullptr;}
}
extern "C" void poremc_query_destroy(void*p){delete static_cast<BoundQuery*>(p);}
extern "C" int poremc_query_energy(void*p,size_t batches,const double*xyz,double*out){
 try{
  if(!p)throw std::invalid_argument("null query");auto& q=*static_cast<BoundQuery*>(p);auto& env=*q.env;
  env.calls.fetch_add(1,std::memory_order_relaxed);
  if(env.m==0||q.atoms==0){std::fill(out,out+batches,0.0);return 0;}
  #pragma omp parallel for schedule(static) num_threads(env.threads) if(batches>=32 && env.threads>1)
  for(std::ptrdiff_t b=0;b<static_cast<std::ptrdiff_t>(batches);++b){
   double total=0;for(size_t a=0;a<q.atoms;++a)total+=env.atom_energy(xyz+(size_t(b)*q.atoms+a)*3,q.sigma[a],q.epsilon[a],q.charge[a],0,0,q.tables[a].get());out[b]=total;
  }return 0;
 }catch(const std::exception&e){last_error=e.what();return -1;}
}

namespace {
struct ShortRangeCuda {
 void* library=nullptr;
 void* handle=nullptr;
 decltype(&poremc_pair_cuda_destroy) destroy=nullptr;
 decltype(&poremc_pair_cuda_error) error=nullptr;
 decltype(&poremc_pair_cuda_device) device=nullptr;
 decltype(&poremc_pair_cuda_upload_guests) upload=nullptr;
 decltype(&poremc_pair_cuda_energy) energy=nullptr;

 ~ShortRangeCuda(){
  if(handle)destroy(handle);
#if defined(__unix__) || defined(__APPLE__)
  if(library)dlclose(library);
#endif
 }

 void open(const Environment& env,const char* path){
#if defined(__unix__) || defined(__APPLE__)
  library=dlopen(path,RTLD_NOW|RTLD_LOCAL);
  if(!library)throw std::runtime_error(dlerror());
  auto create=reinterpret_cast<decltype(&poremc_pair_cuda_create)>(dlsym(library,"poremc_pair_cuda_create"));
  destroy=reinterpret_cast<decltype(destroy)>(dlsym(library,"poremc_pair_cuda_destroy"));
  error=reinterpret_cast<decltype(error)>(dlsym(library,"poremc_pair_cuda_error"));
  device=reinterpret_cast<decltype(device)>(dlsym(library,"poremc_pair_cuda_device"));
  upload=reinterpret_cast<decltype(upload)>(dlsym(library,"poremc_pair_cuda_upload_guests"));
  energy=reinterpret_cast<decltype(energy)>(dlsym(library,"poremc_pair_cuda_energy"));
  if(!create||!destroy||!error||!device||!upload||!energy)
   throw std::runtime_error("CUDA short-range plug-in symbols are missing; rebuild with python native/build.py --cuda");
  int selected=0;
  if(const char* configured=std::getenv("POREMC_CUDA_DEVICE"))selected=std::stoi(configured);
  std::vector<PoreMCCudaPairSite> sites(env.m);
  for(size_t j=0;j<env.m;++j)
   sites[j]={{env.sx[j],env.sy[j],env.sz[j]},env.sigma[j],env.sqrt_epsilon[j],env.charge[j]};
  PoreMCCudaPairHost host{};
  std::copy(env.cell.begin(),env.cell.end(),host.cell);
  std::copy(env.inv.begin(),env.inv.end(),host.inverse);
  std::copy(env.bins.begin(),env.bins.end(),host.bins);
  host.cutoff=env.cutoff;host.alpha=env.alpha;
  host.dsf_value=env.dsf_value;host.dsf_slope=env.dsf_slope;
  host.diagonal=env.diagonal;host.shifted=env.shifted;host.coulomb=env.coulomb;
  host.sites=env.m;host.coordinates=sites.data();
  if(env.expanded){
   host.query_cells=env.adj_start.size()-1;host.neighbour_entries=env.adj.size();
   host.offsets=env.adj_start.data();host.neighbours=env.adj.data();
  }
  handle=create(&host,selected);
  if(!handle)throw std::runtime_error(error());
#else
  (void)env;(void)path;
  throw std::runtime_error("dynamic CUDA short-range loading requires POSIX");
#endif
 }

 int checked(int status) const {
  if(status)last_error=error();
  return status;
 }
};
}

extern "C" {
void* poremc_short_cuda_create(void* environment,const char* path){
 try{
  if(!environment||!path||!*path)throw std::invalid_argument("CUDA short-range requires a host and a plug-in path");
  auto context=std::make_unique<ShortRangeCuda>();
  context->open(*static_cast<Environment*>(environment),path);
  return context.release();
 }catch(const std::exception& error){last_error=error.what();return nullptr;}
}
void poremc_short_cuda_destroy(void* context){delete static_cast<ShortRangeCuda*>(context);}
const char* poremc_short_cuda_device(void* context){
 const auto& gpu=*static_cast<ShortRangeCuda*>(context);
 return gpu.device(gpu.handle);
}
int poremc_short_cuda_upload_guests(void* context,size_t sites,const PoreMCCudaPairSite* guests){
 const auto& gpu=*static_cast<ShortRangeCuda*>(context);
 return gpu.checked(gpu.upload(gpu.handle,sites,guests));
}
int poremc_short_cuda_energy(void* context,size_t batches,size_t atoms,const double* xyz,
                            const double* sigma,const double* epsilon,const double* charge,
                            size_t exclude_begin,size_t exclude_end,int parts,double* out){
 const auto& gpu=*static_cast<ShortRangeCuda*>(context);
 return gpu.checked(gpu.energy(gpu.handle,batches,atoms,xyz,sigma,epsilon,charge,
                               exclude_begin,exclude_end,parts,out));
}
}
