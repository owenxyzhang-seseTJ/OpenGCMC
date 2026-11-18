// Independently derived incrementally maintained Ewald / cubic PME state.
#include "ewald_state.h"
#include "ewald_cuda.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>
#if defined(__unix__) || defined(__APPLE__)
#include <dlfcn.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif
namespace poremc {
namespace {
constexpr double PI=3.1415926535897932384626433832795;
constexpr double CK=167100.94689828736;
using Cx=std::complex<double>;
bool power_two(int n){return n>=8 && (n&(n-1))==0;}
std::array<double,9> invert(const std::array<double,9>&h,double&vol){
 vol=h[0]*(h[4]*h[8]-h[5]*h[7])-h[1]*(h[3]*h[8]-h[5]*h[6])+h[2]*(h[3]*h[7]-h[4]*h[6]);
 if(!(vol>0))throw std::invalid_argument("Ewald requires a positive-volume cell");
 return {(h[4]*h[8]-h[5]*h[7])/vol,(h[2]*h[7]-h[1]*h[8])/vol,(h[1]*h[5]-h[2]*h[4])/vol,
 (h[5]*h[6]-h[3]*h[8])/vol,(h[0]*h[8]-h[2]*h[6])/vol,(h[2]*h[3]-h[0]*h[5])/vol,
 (h[3]*h[7]-h[4]*h[6])/vol,(h[1]*h[6]-h[0]*h[7])/vol,(h[0]*h[4]-h[1]*h[3])/vol};
}
std::array<double,3> wave(const std::array<double,9>&a,int nx,int ny,int nz){
 return {2*PI*(a[0]*nx+a[1]*ny+a[2]*nz),2*PI*(a[3]*nx+a[4]*ny+a[5]*nz),2*PI*(a[6]*nx+a[7]*ny+a[8]*nz)};
}
void fft1(std::vector<Cx>&x,bool inverse){
 const std::size_t n=x.size();
 for(std::size_t i=1,j=0;i<n;++i){std::size_t bit=n>>1;for(;j&bit;bit>>=1)j^=bit;j^=bit;if(i<j)std::swap(x[i],x[j]);}
 for(std::size_t len=2;len<=n;len<<=1){const double theta=(inverse?2.0:-2.0)*PI/len;Cx root(std::cos(theta),std::sin(theta));
  for(std::size_t i=0;i<n;i+=len){Cx w(1,0);for(std::size_t j=0;j<len/2;++j){Cx a=x[i+j],b=x[i+j+len/2]*w;x[i+j]=a+b;x[i+j+len/2]=a-b;w*=root;}}}
 if(inverse)for(auto&v:x)v/=double(n);
}
void fft3(std::vector<Cx>&a,const std::array<int,3>&m,bool inverse){
 const std::size_t nx=m[0],ny=m[1],nz=m[2];std::vector<Cx> line;
 line.resize(nz);for(std::size_t i=0;i<nx;++i)for(std::size_t j=0;j<ny;++j){for(std::size_t k=0;k<nz;++k)line[k]=a[(i*ny+j)*nz+k];fft1(line,inverse);for(std::size_t k=0;k<nz;++k)a[(i*ny+j)*nz+k]=line[k];}
 line.resize(ny);for(std::size_t i=0;i<nx;++i)for(std::size_t k=0;k<nz;++k){for(std::size_t j=0;j<ny;++j)line[j]=a[(i*ny+j)*nz+k];fft1(line,inverse);for(std::size_t j=0;j<ny;++j)a[(i*ny+j)*nz+k]=line[j];}
 line.resize(nx);for(std::size_t j=0;j<ny;++j)for(std::size_t k=0;k<nz;++k){for(std::size_t i=0;i<nx;++i)line[i]=a[(i*ny+j)*nz+k];fft1(line,inverse);for(std::size_t i=0;i<nx;++i)a[(i*ny+j)*nz+k]=line[i];}
}
struct CudaPlugin {
 void* library=nullptr;void* handle=nullptr;
 using Create=void*(*)(std::size_t,const double*,const double*,int);
 using Destroy=void(*)(void*);using Error=const char*(*)();
 using Batch=int(*)(void*,std::size_t,std::size_t,const double*,const double*,const double*,double,double,double,double*);
 Destroy destroy=nullptr;Error error=nullptr;Batch batch=nullptr;
 ~CudaPlugin(){if(handle&&destroy)destroy(handle);
#if defined(__unix__) || defined(__APPLE__)
 if(library)dlclose(library);
#endif
 }
 bool open(const std::string&path,const std::vector<double>&k,const std::vector<double>&w,std::string&reason){
#if defined(__unix__) || defined(__APPLE__)
  library=dlopen(path.c_str(),RTLD_NOW|RTLD_LOCAL);if(!library){const char*e=dlerror();reason=e?e:"could not load CUDA plugin";return false;}
  auto create=reinterpret_cast<Create>(dlsym(library,"poremc_ewald_cuda_create"));
  destroy=reinterpret_cast<Destroy>(dlsym(library,"poremc_ewald_cuda_destroy"));error=reinterpret_cast<Error>(dlsym(library,"poremc_ewald_cuda_last_error"));batch=reinterpret_cast<Batch>(dlsym(library,"poremc_ewald_cuda_batch"));
  if(!create||!destroy||!error||!batch){reason="CUDA plugin ABI symbols are missing";return false;}
  int device=0;if(const char*s=std::getenv("POREMC_CUDA_DEVICE"))device=std::stoi(s);
  handle=create(w.size(),k.data(),w.data(),device);if(!handle){reason=error();return false;}return true;
#else
  (void)path;(void)k;(void)w;reason="dynamic CUDA plugin loading requires POSIX";return false;
#endif
 }
};
}
struct EwaldState::Impl {
 ReciprocalConfig c;double volume=0, self=0,bg=0;
 std::array<double,9> inverse;
 std::vector<double> k,coefficient,green,mesh_charge,mesh_phi,host_mesh_phi;
 struct ModeRow {int i,j;std::size_t begin,end;};
 std::vector<ModeRow> mode_rows;
 std::vector<std::array<int,3>> integer_modes;
 std::vector<Cx> current;
 ChargeContribution host;
 double total_q=0,total_q2=0,host_reciprocal=0,mesh_reciprocal=0;
 mutable ReciprocalStats counters;
 std::unique_ptr<CudaPlugin> cuda;
 std::string backend="cpu";
 Impl(const ReciprocalConfig&config,std::size_t n,const double*xyz,const double*q):c(config){
  if(!(c.alpha>0)||!std::isfinite(c.alpha))throw std::invalid_argument("positive finite Ewald alpha required");
  if(c.method<0||c.method>1||c.backend<0||c.backend>2)throw std::invalid_argument("unknown reciprocal method/backend");
  if(c.k_cutoff<0||!std::isfinite(c.k_cutoff))throw std::invalid_argument("reciprocal k_cutoff must be finite and nonnegative");
  if(c.method==1&&c.k_cutoff!=0)throw std::invalid_argument("radial reciprocal cutoff applies to direct Ewald only");
  inverse=invert(c.cell,volume);self=-CK*c.alpha/std::sqrt(PI);bg=c.background?-CK*PI/(2*c.alpha*c.alpha*volume):0;
  if(c.method==0){
   for(int m:c.kmax)if(m<0)throw std::invalid_argument("nonnegative reciprocal limits required");
   const double cutoff2=c.k_cutoff*c.k_cutoff;
   for(int i=-c.kmax[0];i<=c.kmax[0];++i)for(int j=-c.kmax[1];j<=c.kmax[1];++j){
    const std::size_t begin=coefficient.size();
    for(int l=-c.kmax[2];l<=c.kmax[2];++l){
     if(i==0&&j==0&&l==0)continue;
     auto a=wave(inverse,i,j,l);double k2=a[0]*a[0]+a[1]*a[1]+a[2]*a[2];
     if(c.k_cutoff>0&&k2>cutoff2)continue;
     ++counters.full_modes;
     if(c.half_spectrum&&!(i>0||(i==0&&(j>0||(j==0&&l>0)))))continue;
     k.insert(k.end(),a.begin(),a.end());integer_modes.push_back({i,j,l});
     coefficient.push_back((c.half_spectrum?2.0:1.0)*CK*2*PI/volume*std::exp(-k2/(4*c.alpha*c.alpha))/k2);
    }
    if(coefficient.size()>begin)mode_rows.push_back({i+c.kmax[0],j+c.kmax[1],begin,coefficient.size()});
   }
   if(coefficient.empty())throw std::invalid_argument("at least one Ewald reciprocal mode required");
   current.assign(coefficient.size(),Cx(0,0));counters.modes=coefficient.size();
  }else{
   for(int m:c.mesh)if(!power_two(m))throw std::invalid_argument("PME mesh dimensions must be powers of two >= 8");
   const std::size_t count=std::size_t(c.mesh[0])*c.mesh[1]*c.mesh[2];coefficient.resize(count);mesh_charge.assign(count,0);mesh_phi.assign(count,0);counters.mesh_points=count;
   std::vector<Cx> kernel(count);
   for(int i=0;i<c.mesh[0];++i)for(int j=0;j<c.mesh[1];++j)for(int l=0;l<c.mesh[2];++l){
    int ni=i<=c.mesh[0]/2?i:i-c.mesh[0],nj=j<=c.mesh[1]/2?j:j-c.mesh[1],nl=l<=c.mesh[2]/2?l:l-c.mesh[2];
    std::size_t index=(std::size_t(i)*c.mesh[1]+j)*c.mesh[2]+l;
    // Omit Nyquist planes. This preserves a real symmetric Green operator also
    // in triclinic cells, where a representative Nyquist sign is ambiguous.
    if((ni==0&&nj==0&&nl==0)||std::abs(ni)==c.mesh[0]/2||std::abs(nj)==c.mesh[1]/2||std::abs(nl)==c.mesh[2]/2)continue;
    auto a=wave(inverse,ni,nj,nl);double k2=a[0]*a[0]+a[1]*a[1]+a[2]*a[2];
    double b=(2+std::cos(2*PI*ni/c.mesh[0]))/3*(2+std::cos(2*PI*nj/c.mesh[1]))/3*(2+std::cos(2*PI*nl/c.mesh[2]))/3;
    coefficient[index]=CK*2*PI/volume*std::exp(-k2/(4*c.alpha*c.alpha))/k2/(b*b);kernel[index]=coefficient[index];
   }
   fft3(kernel,c.mesh,true);++counters.fft_transforms;green.resize(count);for(std::size_t i=0;i<count;++i)green[i]=count*kernel[i].real();
  }
  host=make(n,xyz,q);total_q=host.charge;total_q2=host.squared_charge;
  if(c.method==0){current=host.modes;host_reciprocal=reciprocal(current);}else{for(std::size_t i=0;i<host.indices.size();++i)mesh_charge[host.indices[i]]+=host.values[i];refresh_mesh();host_mesh_phi=mesh_phi;host_reciprocal=mesh_reciprocal;}
  counters.host_builds=1;
  if(c.backend && c.method==0){
   cuda=std::make_unique<CudaPlugin>();std::string reason,path=c.cuda_library;
   if(path.empty()){const char*env=std::getenv("POREMC_EWALD_CUDA_LIBRARY");path=env?env:"libporemc_ewald_cuda.so";}
   if(cuda->open(path,k,coefficient,reason))backend="cuda";
   else{cuda.reset();if(c.backend==2)throw std::runtime_error("required CUDA reciprocal backend unavailable: "+reason);backend="cpu (CUDA unavailable)";}
  }else if(c.backend==2&&c.method==1)throw std::invalid_argument("PME currently uses native CPU FFT; CUDA reciprocal batching is available for direct Ewald");
 }
 ChargeContribution make(std::size_t atoms,const double*xyz,const double*q)const{
  ChargeContribution r;
  for(std::size_t a=0;a<atoms;++a){r.charge+=q[a];r.squared_charge+=q[a]*q[a];}
  if(c.method==0){
   r.modes.assign(coefficient.size(),Cx(0,0));
   std::array<std::vector<Cx>,3> powers;
   for(int d=0;d<3;++d)powers[d].resize(2*c.kmax[d]+1);
   for(std::size_t a=0;a<atoms;++a)if(q[a]){
    // Only three fundamental trigonometric evaluations per atom. All integer
    // reciprocal harmonics follow by complex recurrence; negative harmonics
    // use exact conjugation. Mode rows retain the configured rectangular or
    // radial subset; conjugate pairing changes storage, never its energy.
    for(int d=0;d<3;++d){
     double f=xyz[3*a]*inverse[d]+xyz[3*a+1]*inverse[3+d]+xyz[3*a+2]*inverse[6+d];
     f-=std::floor(f);double phase=2*PI*f;Cx unit(std::cos(phase),std::sin(phase));
     int middle=c.kmax[d];powers[d][middle]=Cx(1,0);
     for(int h=1;h<=middle;++h){powers[d][middle+h]=powers[d][middle+h-1]*unit;powers[d][middle-h]=std::conj(powers[d][middle+h]);}
    }
    for(const auto& row:mode_rows){
     Cx xy=q[a]*powers[0][row.i]*powers[1][row.j];
     for(std::size_t t=row.begin;t<row.end;++t)r.modes[t]+=xy*powers[2][integer_modes[t][2]+c.kmax[2]];
    }
   }
  }
  else{
   std::vector<std::pair<std::size_t,double>> terms;terms.reserve(atoms*64);
   for(std::size_t a=0;a<atoms;++a)if(q[a]){
    int nodes[3][4];double weights[3][4];
    for(int d=0;d<3;++d){double f=xyz[3*a]*inverse[d]+xyz[3*a+1]*inverse[3+d]+xyz[3*a+2]*inverse[6+d];f-=std::floor(f);double u=f*c.mesh[d];int base=int(std::floor(u));double t=u-base,one=1-t;
     weights[d][0]=one*one*one/6;weights[d][1]=(3*t*t*t-6*t*t+4)/6;weights[d][2]=(-3*t*t*t+3*t*t+3*t+1)/6;weights[d][3]=t*t*t/6;
     for(int b=0;b<4;++b)nodes[d][b]=(base+b-1+c.mesh[d])%c.mesh[d];}
    for(int i=0;i<4;++i)for(int j=0;j<4;++j)for(int l=0;l<4;++l){double v=q[a]*weights[0][i]*weights[1][j]*weights[2][l];if(v)terms.emplace_back((std::size_t(nodes[0][i])*c.mesh[1]+nodes[1][j])*c.mesh[2]+nodes[2][l],v);}
   }
   std::stable_sort(terms.begin(),terms.end(),[](const auto&a,const auto&b){return a.first<b.first;});
   for(const auto&t:terms){if(!r.indices.empty()&&r.indices.back()==t.first)r.values.back()+=t.second;else{r.indices.push_back(t.first);r.values.push_back(t.second);}}
  }
  return r;
 }
 double reciprocal(const std::vector<Cx>&s)const{double e=0;for(std::size_t i=0;i<s.size();++i)e+=coefficient[i]*std::norm(s[i]);return e;}
 std::size_t difference(std::size_t a,std::size_t b)const{
  const std::size_t nz=c.mesh[2],ny=c.mesh[1];
  std::size_t az=a%nz,bz=b%nz;a/=nz;b/=nz;std::size_t ay=a%ny,by=b%ny;std::size_t ax=a/ny,bx=b/ny;
  // Dimensions are powers of two, so wrapping is exact unsigned masking.
  return (((ax-bx)&(c.mesh[0]-1))*ny+((ay-by)&(ny-1)))*nz+((az-bz)&(nz-1));
 }
 double bilinear(const ChargeContribution&a,const ChargeContribution&b)const{
  double u=0;for(std::size_t i=0;i<a.indices.size();++i){double e=0;for(std::size_t j=0;j<b.indices.size();++j)e+=green[difference(a.indices[i],b.indices[j])]*b.values[j];u+=a.values[i]*e;}return u;
 }
 double quadratic(const ChargeContribution&a)const{
  double u=0;for(std::size_t i=0;i<a.indices.size();++i){u+=a.values[i]*a.values[i]*green[0];double v=0;for(std::size_t j=0;j<i;++j)v+=a.values[j]*green[difference(a.indices[i],a.indices[j])];u+=2*a.values[i]*v;}return u;
 }
 double delta(const ChargeContribution&a,const ChargeContribution*ex,const ChargeContribution*pre)const{
  double energy=self*a.squared_charge;double baseq=total_q-(ex?ex->charge:0)+(pre?pre->charge:0);energy+=bg*(2*baseq*a.charge+a.charge*a.charge);
  if(c.method==0){for(std::size_t i=0;i<coefficient.size();++i){Cx base=current[i];if(ex&&!ex->modes.empty())base-=ex->modes[i];if(pre&&!pre->modes.empty())base+=pre->modes[i];Cx d=a.modes.empty()?Cx(0,0):a.modes[i];energy+=coefficient[i]*(2*std::real(std::conj(base)*d)+std::norm(d));}}
  else{for(std::size_t i=0;i<a.indices.size();++i)energy+=a.values[i]*mesh_phi[a.indices[i]];energy+=quadratic(a);if(ex)energy-=2*bilinear(a,*ex);if(pre)energy+=2*bilinear(a,*pre);}
  return energy;
 }
 double host_cross(const ChargeContribution&a)const{
  double u=0;
  if(c.method==0){if(!a.modes.empty())for(std::size_t i=0;i<coefficient.size();++i)u+=2*coefficient[i]*std::real(std::conj(host.modes[i])*a.modes[i]);}
  else for(std::size_t i=0;i<a.indices.size();++i)u+=a.values[i]*host_mesh_phi[a.indices[i]];
  return u;
 }
 double host_mesh_potential(const double*xyz)const{
  int nodes[3][4];double weights[3][4];
  for(int d=0;d<3;++d){double f=xyz[0]*inverse[d]+xyz[1]*inverse[3+d]+xyz[2]*inverse[6+d];f-=std::floor(f);double u=f*c.mesh[d];int base=int(std::floor(u));double t=u-base,one=1-t;
   weights[d][0]=one*one*one/6;weights[d][1]=(3*t*t*t-6*t*t+4)/6;weights[d][2]=(-3*t*t*t+3*t*t+3*t+1)/6;weights[d][3]=t*t*t/6;
   for(int b=0;b<4;++b)nodes[d][b]=(base+b-1+c.mesh[d])%c.mesh[d];}
  double result=0;
  for(int i=0;i<4;++i)for(int j=0;j<4;++j)for(int l=0;l<4;++l)result+=weights[0][i]*weights[1][j]*weights[2][l]*host_mesh_phi[(std::size_t(nodes[0][i])*c.mesh[1]+nodes[1][j])*c.mesh[2]+nodes[2][l]];
  return result;
 }
 void refresh_mesh(){
  std::vector<Cx> work(mesh_charge.begin(),mesh_charge.end());fft3(work,c.mesh,false);++counters.fft_transforms;
  for(std::size_t i=0;i<work.size();++i)work[i]*=coefficient[i];
  fft3(work,c.mesh,true);++counters.fft_transforms;
  mesh_reciprocal=0;for(std::size_t i=0;i<work.size();++i){mesh_phi[i]=2*work.size()*work[i].real();mesh_reciprocal+=.5*mesh_charge[i]*mesh_phi[i];}
 }
};
EwaldState::EwaldState(const ReciprocalConfig&c,std::size_t n,const double*x,const double*q):p_(new Impl(c,n,x,q)){}
EwaldState::~EwaldState()=default;
ChargeContribution EwaldState::contribution(std::size_t n,const double*x,const double*q)const{++p_->counters.contributions;return p_->make(n,x,q);}
ChargeContribution EwaldState::translated_contribution(const ChargeContribution&old,const std::array<double,3>&d)const{
 if(p_->c.method!=0)throw std::invalid_argument("translated reciprocal contributions require direct Ewald");
 if(old.modes.size()!=p_->coefficient.size())throw std::invalid_argument("translated reciprocal contribution has incompatible modes");
 const double unit=1.;auto result=p_->make(1,d.data(),&unit);
 for(std::size_t k=0;k<result.modes.size();++k)result.modes[k]*=old.modes[k];
 result.charge=old.charge;result.squared_charge=old.squared_charge;
 ++p_->counters.contributions;return result;
}
void EwaldState::accumulate(ChargeContribution&t,const ChargeContribution&s,double sign)const{
 t.charge+=sign*s.charge;t.squared_charge+=sign*s.squared_charge;
 if(p_->c.method==0){if(t.modes.empty())t.modes.assign(p_->coefficient.size(),Cx(0,0));if(!s.modes.empty())for(std::size_t i=0;i<t.modes.size();++i)t.modes[i]+=sign*s.modes[i];}
 else{
  std::vector<std::size_t> ix;std::vector<double> vals;ix.reserve(t.indices.size()+s.indices.size());vals.reserve(ix.capacity());std::size_t i=0,j=0;
  while(i<t.indices.size()||j<s.indices.size()){
   if(j==s.indices.size()||(i<t.indices.size()&&t.indices[i]<s.indices[j])){ix.push_back(t.indices[i]);vals.push_back(t.values[i]);++i;}
   else if(i==t.indices.size()||s.indices[j]<t.indices[i]){ix.push_back(s.indices[j]);vals.push_back(sign*s.values[j]);++j;}
   else{ix.push_back(t.indices[i]);vals.push_back(t.values[i]+sign*s.values[j]);++i;++j;}}
  t.indices.swap(ix);t.values.swap(vals);
 }
}
double EwaldState::insertion_energy(const ChargeContribution&a,const ChargeContribution*ex,const ChargeContribution*pre)const{return p_->delta(a,ex,pre);}
double EwaldState::host_insertion_energy(const ChargeContribution&a)const{++p_->counters.host_field_evaluations;return p_->host_cross(a);}
double EwaldState::host_insertion_energy(std::size_t n,const double*x,const double*q)const{
 ++p_->counters.host_field_evaluations;
 if(p_->c.method==0)return p_->host_cross(p_->make(n,x,q));
 double result=0;for(std::size_t a=0;a<n;++a)if(q[a])result+=q[a]*p_->host_mesh_potential(x+3*a);return result;
}
std::vector<double>EwaldState::host_potential_grid(const std::array<int,3>&m)const{
 for(int n:m)if(!power_two(n))throw std::invalid_argument("host Fourier grid dimensions must be powers of two >= 8");
 const std::size_t count=std::size_t(m[0])*m[1]*m[2];std::vector<double> result(count);
 if(p_->c.method==0){
  std::vector<Cx> spectrum(count,Cx(0,0));
  auto bin=[&](int i,int j,int l){return (std::size_t((i%m[0]+m[0])%m[0])*m[1]+(j%m[1]+m[1])%m[1])*m[2]+(l%m[2]+m[2])%m[2];};
  for(std::size_t t=0;t<p_->coefficient.size();++t){
   const auto&n=p_->integer_modes[t];
   // S_H uses exp(+ikr), whereas the physical potential uses exp(+ikr)
   // times conj(S_H).  An inverse FFT has this sign and a 1/count factor.
   // Scatter-add, rather than assignment, preserves exact values at nodes
   // even when high frequencies alias into the chosen grid.
   const Cx a=double(count)*(p_->c.half_spectrum?1.0:2.0)*p_->coefficient[t]*std::conj(p_->host.modes[t]);
   spectrum[bin(n[0],n[1],n[2])]+=a;
   if(p_->c.half_spectrum)spectrum[bin(-n[0],-n[1],-n[2])]+=std::conj(a);
  }
  fft3(spectrum,m,true);++p_->counters.fft_transforms;
  for(std::size_t i=0;i<count;++i)result[i]=spectrum[i].real();
 }else{
  const auto&h=p_->c.cell;
  for(int i=0;i<m[0];++i)for(int j=0;j<m[1];++j)for(int k=0;k<m[2];++k){
   const double f[3]={double(i)/m[0],double(j)/m[1],double(k)/m[2]};
   const double x[3]={f[0]*h[0]+f[1]*h[3]+f[2]*h[6],f[0]*h[1]+f[1]*h[4]+f[2]*h[7],f[0]*h[2]+f[1]*h[5]+f[2]*h[8]};
   result[(std::size_t(i)*m[1]+j)*m[2]+k]=p_->host_mesh_potential(x);
  }
 }
 ++p_->counters.host_grid_builds;return result;
}
void EwaldState::batch_insertion(std::size_t B,std::size_t A,const double*x,const double*q,const ChargeContribution*ex,const ChargeContribution*pre,double*out)const{
 ++p_->counters.candidate_batches;p_->counters.candidates+=B;
 if(!B)return;
 if(p_->cuda){
  std::vector<double> base(2*p_->current.size());for(std::size_t i=0;i<p_->current.size();++i){Cx s=p_->current[i];if(ex&&!ex->modes.empty())s-=ex->modes[i];if(pre&&!pre->modes.empty())s+=pre->modes[i];base[2*i]=s.real();base[2*i+1]=s.imag();}
  double baseq=p_->total_q-(ex?ex->charge:0)+(pre?pre->charge:0);
  if(p_->cuda->batch(p_->cuda->handle,B,A,x,q,base.data(),baseq,p_->self,p_->bg,out))throw std::runtime_error(p_->cuda->error());
  ++p_->counters.gpu_batches;
 }else{
#ifdef _OPENMP
  #pragma omp parallel for if(B>=8 && p_->c.threads>1) num_threads(p_->c.threads) schedule(static)
#endif
  for(std::int64_t b=0;b<static_cast<std::int64_t>(B);++b){auto a=p_->make(A,x+std::size_t(b)*A*3,q);out[b]=p_->delta(a,ex,pre);}
 }
}
void EwaldState::commit(const ChargeContribution*a,const ChargeContribution*r){
 if(!a&&!r)return;
 p_->total_q+=(a?a->charge:0)-(r?r->charge:0);p_->total_q2+=(a?a->squared_charge:0)-(r?r->squared_charge:0);
 if(p_->c.method==0){for(std::size_t i=0;i<p_->current.size();++i){if(r&&!r->modes.empty())p_->current[i]-=r->modes[i];if(a&&!a->modes.empty())p_->current[i]+=a->modes[i];}}
 else{if(r)for(std::size_t i=0;i<r->indices.size();++i)p_->mesh_charge[r->indices[i]]-=r->values[i];if(a)for(std::size_t i=0;i<a->indices.size();++i)p_->mesh_charge[a->indices[i]]+=a->values[i];p_->refresh_mesh();}
 ++p_->counters.accepted_commits;
}
double EwaldState::total_host_relative_energy()const{
 double reciprocal=p_->c.method==0?p_->reciprocal(p_->current):p_->mesh_reciprocal;
 return reciprocal-p_->host_reciprocal+p_->self*(p_->total_q2-p_->host.squared_charge)+p_->bg*(p_->total_q*p_->total_q-p_->host.charge*p_->host.charge);
}
std::string EwaldState::serialize()const{
 std::ostringstream s;s<<std::setprecision(std::numeric_limits<double>::max_digits10);const auto n=p_->c.method==0?p_->current.size():p_->mesh_charge.size();
 s<<"PoreMC-EwaldState-v2 "<<p_->c.method<<' '<<n<<' '<<p_->total_q<<' '<<p_->total_q2<<' '<<p_->c.half_spectrum<<' '<<p_->c.k_cutoff<<'\n';
 if(p_->c.method==0)for(const auto&z:p_->current)s<<z.real()<<' '<<z.imag()<<'\n';else for(double v:p_->mesh_charge)s<<v<<'\n';
 s<<p_->counters.host_builds<<' '<<p_->counters.contributions<<' '<<p_->counters.candidate_batches<<' '<<p_->counters.candidates<<' '<<p_->counters.accepted_commits<<' '<<p_->counters.fft_transforms<<' '<<p_->counters.gpu_batches<<' '<<p_->counters.host_grid_builds<<' '<<p_->counters.host_field_evaluations<<'\n';return s.str();
}
void EwaldState::restore(const std::string&data){
 std::istringstream s(data);std::string version;int method;std::size_t n;double q,q2,cutoff;bool half;s>>version>>method>>n>>q>>q2>>half>>cutoff;
 std::size_t expected=p_->c.method==0?p_->current.size():p_->mesh_charge.size();
 if(!s||version!="PoreMC-EwaldState-v2"||method!=p_->c.method||n!=expected||half!=p_->c.half_spectrum||cutoff!=p_->c.k_cutoff)throw std::invalid_argument("reciprocal checkpoint format/configuration mismatch");
 std::vector<Cx> modes;std::vector<double> mesh;
 if(method==0){modes.resize(n);for(auto&z:modes){double re,im;s>>re>>im;if(!s||!std::isfinite(re)||!std::isfinite(im))throw std::invalid_argument("invalid reciprocal checkpoint value");z=Cx(re,im);}}
 else{mesh.resize(n);for(double&v:mesh){s>>v;if(!s||!std::isfinite(v))throw std::invalid_argument("invalid mesh checkpoint value");}}
 ReciprocalStats counts=p_->counters;s>>counts.host_builds>>counts.contributions>>counts.candidate_batches>>counts.candidates>>counts.accepted_commits>>counts.fft_transforms>>counts.gpu_batches>>counts.host_grid_builds>>counts.host_field_evaluations;
 if(!s||!std::isfinite(q)||!std::isfinite(q2))throw std::invalid_argument("truncated reciprocal checkpoint");
 p_->total_q=q;p_->total_q2=q2;if(method==0)p_->current.swap(modes);else{p_->mesh_charge.swap(mesh);p_->refresh_mesh();}p_->counters=counts;
}
ReciprocalStats EwaldState::stats()const{return p_->counters;}
const char*EwaldState::backend_name()const{return p_->backend.c_str();}
const ReciprocalConfig&EwaldState::config()const{return p_->c;}
}
