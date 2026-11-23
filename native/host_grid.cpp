// Immutable periodic host fields for reference-potential CBMC.
// The physical target is corrected separately; no physical energy is capped.
#include "host_grid.h"
#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>
#ifdef _WIN32
#define NOMINMAX
#include <filesystem>
#include <io.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

extern "C" {
void* poremc_query_create(void*,std::size_t,const double*,const double*,const double*);
void poremc_query_destroy(void*);
int poremc_query_energy(void*,std::size_t,const double*,double*);
const char* poremc_last_error();
}

namespace poremc {
namespace {
using Query=std::unique_ptr<void,void(*)(void*)>;
Query make_query(void* env,double s,double e,double q) {
  Query result(poremc_query_create(env,1,&s,&e,&q),poremc_query_destroy);
  if(!result)throw std::runtime_error(poremc_last_error());
  return result;
}
std::array<double,4> cubic(double t) {
  // Cardinal cubic with central finite-difference tangents (Catmull-Rom).
  const double t2=t*t,t3=t2*t;
  return {{-.5*t+t2-.5*t3,1-2.5*t2+1.5*t3,.5*t+2*t2-1.5*t3,-.5*t2+.5*t3}};
}
std::uint64_t checksum(const void* ptr,std::size_t bytes,std::uint64_t h=14695981039346656037ull) {
  const auto* p=static_cast<const unsigned char*>(ptr);
  for(std::size_t i=0;i<bytes;++i){h^=p[i];h*=1099511628211ull;}
  return h;
}
template<class T> bool read_scalar(std::istream& s,T& x) {
  return bool(s.read(reinterpret_cast<char*>(&x),sizeof(T)));
}
// The accepted PMCHG061 layout is independent of the C++ stream ABI.  Use
// an exclusively created sibling file so an incomplete or racing writer
// never exposes its payload at the requested path.
class GridCacheWriter {
 public:
  explicit GridCacheWriter(const std::string& destination) : path_(destination) {
#ifdef _WIN32
    auto target = std::filesystem::u8path(path_);
    auto parent = target.has_parent_path() ? target.parent_path() : std::filesystem::path(L".");
    std::filesystem::create_directories(parent);
    wchar_t name[MAX_PATH];
    if (!GetTempFileNameW(parent.c_str(), L"phg", 0, name))
      throw std::runtime_error("cannot create temporary host grid in " + path_);
    temporary_ = std::filesystem::path(name).u8string();
    stream_ = _wfopen(name, L"wb");
    if (!stream_) {
      const int error = errno;
      DeleteFileW(name);
      fail("open", error);
    }
#else
    const auto slash = path_.find_last_of('/');
    const std::string directory = slash == std::string::npos ? "" : path_.substr(0, slash + 1);
    // Preserve the native API's recursive directory creation.  Avoid
    // std::filesystem on the Linux HPC path, including the rename step.
    for (size_t i = 1; i < directory.size(); ++i) {
      if (directory[i] == '/' && directory[i - 1] != '/') {
        const std::string component = directory.substr(0, i);
        if (::mkdir(component.c_str(), 0777) != 0 && errno != EEXIST)
          fail("create parent directory", errno);
      }
    }
    temporary_ = directory + ".poremc-hg-XXXXXX";
    int fd = ::mkstemp(temporary_.data());
    if (fd < 0) fail("create temporary file", errno);
    stream_ = ::fdopen(fd, "wb");
    if (!stream_) {
      const int error = errno;
      ::close(fd);
      ::unlink(temporary_.c_str());
      fail("open", error);
    }
#endif
  }

  GridCacheWriter(const GridCacheWriter&) = delete;
  GridCacheWriter& operator=(const GridCacheWriter&) = delete;

  ~GridCacheWriter() {
    if (stream_) std::fclose(stream_);
    if (!temporary_.empty()) {
#ifdef _WIN32
      DeleteFileW(std::filesystem::u8path(temporary_).c_str());
#else
      ::unlink(temporary_.c_str());
#endif
    }
  }

  void write(const void* source, size_t bytes, std::uint64_t* hash = nullptr) {
    const auto* data = static_cast<const unsigned char*>(source);
    while (bytes) {
      const size_t n = std::min(bytes, size_t(1) << 20);
      if (std::fwrite(data, 1, n, stream_) != n) fail("write", errno);
      if (hash) *hash = checksum(data, n, *hash);
      data += n;
      bytes -= n;
    }
  }

  template<class T> void scalar(T value) { write(&value, sizeof value); }

  void publish() {
    if (std::fflush(stream_) != 0) fail("flush", errno);
#ifdef _WIN32
    if (::_commit(::_fileno(stream_)) != 0) fail("sync", errno);
#else
    int rc;
    do { rc = ::fsync(::fileno(stream_)); } while (rc != 0 && errno == EINTR);
    if (rc != 0) fail("sync", errno);
#endif
    auto* closing = std::exchange(stream_, nullptr);
    if (std::fclose(closing) != 0) fail("close", errno);
#ifdef _WIN32
    if (!MoveFileExW(std::filesystem::u8path(temporary_).c_str(),
                     std::filesystem::u8path(path_).c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
      throw std::runtime_error("host grid cache rename failed: " + path_ +
                               " (Windows error " + std::to_string(GetLastError()) + ")");
#else
    if (::rename(temporary_.c_str(), path_.c_str()) != 0) fail("rename", errno);
#endif
    temporary_.clear();
  }

 private:
  [[noreturn]] void fail(const char* operation, int code) const {
    throw std::runtime_error("host grid cache " + std::string(operation) + " failed for " +
                             path_ + ": " + std::strerror(code ? code : EIO));
  }
  const std::string& path_;
  std::string temporary_;
  std::FILE* stream_ = nullptr;
};

// The lock belongs to the OS file handle and is released on process exit.
// Keep the lock file in place to prevent waiters from locking different inodes.
class FieldLock {
public:
  explicit FieldLock(const std::string& path) {
#ifdef _WIN32
    const auto file=std::filesystem::u8path(path);
    handle_=CreateFileW(file.c_str(),GENERIC_READ|GENERIC_WRITE,FILE_SHARE_READ|FILE_SHARE_WRITE,
                        nullptr,OPEN_ALWAYS,FILE_ATTRIBUTE_NORMAL,nullptr);
    if(handle_==INVALID_HANDLE_VALUE)throw std::runtime_error("cannot open field lock: "+path);
    if(!LockFileEx(handle_,LOCKFILE_EXCLUSIVE_LOCK,0,1,0,&region_)){
      CloseHandle(handle_);
      throw std::runtime_error("cannot acquire field lock: "+path);
    }
#else
    fd_=::open(path.c_str(),O_CREAT|O_RDWR,0666);
    if(fd_<0)throw std::runtime_error("cannot open field lock: "+path);
    int result;
    do{result=::flock(fd_,LOCK_EX);}while(result<0&&errno==EINTR);
    if(result<0){::close(fd_);throw std::runtime_error("cannot acquire field lock: "+path);}
#endif
  }
  FieldLock(const FieldLock&)=delete;
  FieldLock& operator=(const FieldLock&)=delete;
  ~FieldLock(){
#ifdef _WIN32
    CloseHandle(handle_);
#else
    ::close(fd_);
#endif
  }
private:
#ifdef _WIN32
  HANDLE handle_;
  OVERLAPPED region_{};
#else
  int fd_;
#endif
};
}

struct HostGrid::Impl {
  struct Field {std::vector<double> values;std::vector<std::uint8_t> bad;};
  struct Type {
    double sigma,epsilon,charge;
    std::size_t lj;
    Query exact{nullptr,poremc_query_destroy};
    std::vector<std::uint8_t> steep;
    // Derived at construction from the shareable raw fields. A one-node
    // left and two-node right periodic halo make each 4x4x4 stencil contiguous in z.
    std::vector<double> packed_energy, packed_weight;
  };
  HostGridConfig cfg;
  std::array<double,9> inv{};
  std::size_t points=0;
  std::vector<std::array<double,2>> lj_parameters;
  std::vector<Field> lj;
  Field phi;
  std::vector<Type> types;
  std::vector<std::size_t> site_type;
  HostReciprocalEnergy reciprocal_exact;
  mutable HostGridStats statistics;

  Impl(const HostGridConfig& c,void* host,std::size_t atoms,const double* s,
       const double* e,const double* q,HostReciprocalGridBuilder builder,
       HostReciprocalEnergy exact)
    :cfg(c),reciprocal_exact(std::move(exact)) {
    if(!host||!atoms)throw std::invalid_argument("host grid requires a host and molecular sites");
    if(!(cfg.steep_limit>0)||!std::isfinite(cfg.steep_limit)||cfg.surrogate_cap<0||cfg.threads<1||
       !std::isfinite(cfg.surrogate_cap))throw std::invalid_argument("invalid host grid reference limits");
    if(cfg.steep_interpolation<0||cfg.steep_interpolation>1||!(cfg.temperature>0)||!std::isfinite(cfg.temperature))
      throw std::invalid_argument("invalid host grid Boltzmann interpolation configuration");
    if(!cfg.cache_path.empty()&&cfg.cache_key.empty())throw std::invalid_argument("host grid cache requires a physics fingerprint");
    points=1;
    for (int m : cfg.mesh) {
      if (m < 4) throw std::invalid_argument("host grid dimensions must be at least four");
      if (points > std::vector<double>().max_size() / std::size_t(m))
        throw std::length_error("host grid mesh exceeds addressable field size");
      points *= std::size_t(m);
    }
    const double* h=cfg.cell.data();
    const double v=h[0]*(h[4]*h[8]-h[5]*h[7])-h[1]*(h[3]*h[8]-h[5]*h[6])+h[2]*(h[3]*h[7]-h[4]*h[6]);
    if(!(v>0)||!std::isfinite(v))throw std::invalid_argument("invalid host grid cell");
    inv={{(h[4]*h[8]-h[5]*h[7])/v,(h[2]*h[7]-h[1]*h[8])/v,(h[1]*h[5]-h[2]*h[4])/v,
          (h[5]*h[6]-h[3]*h[8])/v,(h[0]*h[8]-h[2]*h[6])/v,(h[2]*h[3]-h[0]*h[5])/v,
          (h[3]*h[7]-h[4]*h[6])/v,(h[1]*h[6]-h[0]*h[7])/v,(h[0]*h[4]-h[1]*h[3])/v}};
    std::map<std::array<double,2>,std::size_t> lj_map;
    std::map<std::array<double,3>,std::size_t> type_map;
    bool charged=false;
    for(std::size_t i=0;i<atoms;++i){
      auto pair=std::array<double,2>{{s[i],e[i]}};
      auto inserted=lj_map.emplace(pair,lj_parameters.size());
      if(inserted.second)lj_parameters.push_back(pair);
      auto key=std::array<double,3>{{s[i],e[i],q[i]}};
      auto ti=type_map.emplace(key,types.size());
      if(ti.second){
        Type type{s[i],e[i],q[i],inserted.first->second};
        type.exact=make_query(host,s[i],e[i],q[i]);
        types.push_back(std::move(type));
      }
      site_type.push_back(ti.first->second);charged=charged||q[i]!=0.;
    }
    lj.resize(lj_parameters.size());
    for(auto& f:lj){f.values.resize(points);f.bad.resize(points);}
    phi.values.resize(points);phi.bad.resize(points);
    if(!cfg.lj_prefix.empty()){
      for(std::size_t i=0;i<lj.size();++i){
        const auto p=lj_parameters[i];
        if(p[1]==0.)continue;
        std::uint64_t sigma=0,epsilon=0;
        std::memcpy(&sigma,&p[0],sizeof(sigma));
        std::memcpy(&epsilon,&p[1],sizeof(epsilon));
        std::ostringstream suffix;
        suffix<<'-'<<std::setfill('0')<<std::hex<<std::setw(16)<<sigma<<'-'<<std::setw(16)<<epsilon;
        raw_field(host,lj[i],p[0],p[1],0.,cfg.lj_prefix+suffix.str()+".bin",{});
      }
      if(charged&&!cfg.electrostatic_path.empty())
        raw_field(host,phi,0.,0.,1.,cfg.electrostatic_path,std::move(builder));
      statistics.builds=statistics.raw_field_builds!=0;
      statistics.cache_writes=statistics.raw_field_builds;
      statistics.cache_loads=statistics.raw_field_loads;
    }
    else if(load())++statistics.cache_loads;
    else {
      build(host,charged,std::move(builder));++statistics.builds;
      if(!cfg.cache_path.empty()){save();++statistics.cache_writes;}
    }
    make_stencils();
    statistics.mesh_points=points;statistics.lj_fields=lj.size();statistics.site_types=types.size();
    statistics.bytes+=std::uint64_t((lj.size()+1)*points*(sizeof(double)+sizeof(std::uint8_t))+
                                   types.size()*points*sizeof(std::uint8_t));
  }

  std::size_t index(int x,int y,int z) const {
    return (std::size_t(x)*cfg.mesh[1]+y)*cfg.mesh[2]+z;
  }
  int periodic(int x,int d) const {
    const int m=cfg.mesh[d];return x<0?x+m:(x>=m?x-m:x);
  }
  bool read_field(const std::string& path,Field& field) const {
    std::ifstream in(path,std::ios::binary);
    char magic[8];
    std::uint64_t n=0,expected=0;
    const std::string key=path.substr(path.find_last_of("/\\")+1);
    std::string stored(key.size(),'\0');
    if(!in.read(magic,8)||std::memcmp(magic,"OGCMCF01",8)||
       !in.read(stored.data(),stored.size())||stored!=key||!read_scalar(in,n)||n!=points||
       !in.read(reinterpret_cast<char*>(field.values.data()),points*sizeof(double))||
       !in.read(reinterpret_cast<char*>(field.bad.data()),points)||
       !read_scalar(in,expected)||in.peek()!=std::char_traits<char>::eof())return false;
    const auto h=checksum(field.values.data(),points*sizeof(double));
    return checksum(field.bad.data(),points,h)==expected;
  }
  void raw_field(void* host,Field& field,double s,double e,double q,const std::string& path,
                 HostReciprocalGridBuilder reciprocal) {
    if(read_field(path,field)){++statistics.raw_field_loads;return;}
    FieldLock lock(path+".lock");
    if(read_field(path,field)){++statistics.raw_field_loads;return;}
    auto query=make_query(host,s,e,q);
    const auto spectrum=reciprocal?reciprocal():std::vector<double>{};
    if(reciprocal&&spectrum.size()!=points)throw std::invalid_argument("host reciprocal grid shape mismatch");
    constexpr std::size_t chunk=4096;
    std::vector<double> xyz(3*std::min(chunk,points)), values(std::min(chunk,points));
    for(std::size_t begin=0;begin<points;begin+=chunk){
      const auto n=std::min(chunk,points-begin);
      grid_coordinates(begin,n,xyz.data());
      if(poremc_query_energy(query.get(),n,xyz.data(),values.data()))throw std::runtime_error(poremc_last_error());
      for(std::size_t j=0;j<n;++j){
        const double u=values[j]+(spectrum.empty()?0.:spectrum[begin+j]);
        field.bad[begin+j]=!std::isfinite(u);
        field.values[begin+j]=std::isfinite(u)?u:0.;
      }
    }
    const std::string key=path.substr(path.find_last_of("/\\")+1);
    GridCacheWriter out(path);
    out.write("OGCMCF01",8);
    out.write(key.data(),key.size());
    out.scalar(std::uint64_t(points));
    std::uint64_t h=14695981039346656037ull;
    out.write(field.values.data(),points*sizeof(double),&h);
    out.write(field.bad.data(),points,&h);
    out.scalar(h);
    out.publish();
    ++statistics.raw_field_builds;
  }
  void grid_coordinates(std::size_t begin,std::size_t count,double* xyz) const {
    for(std::size_t j=0;j<count;++j){
      const std::size_t key=begin+j;
      const double fz=double(key%cfg.mesh[2])/cfg.mesh[2];
      const double fy=double((key/cfg.mesh[2])%cfg.mesh[1])/cfg.mesh[1];
      const double fx=double(key/(std::size_t(cfg.mesh[1])*cfg.mesh[2]))/cfg.mesh[0];
      for(int d=0;d<3;++d)xyz[3*j+d]=fx*cfg.cell[d]+fy*cfg.cell[3+d]+fz*cfg.cell[6+d];
    }
  }
  void build(void* host,bool charged,HostReciprocalGridBuilder builder) {
    // One reusable scalar Coulomb field and one LJ field per unique LJ type.
    // Queries are chunked, letting the existing native host cell lists and
    // OpenMP batches perform short-range preprocessing efficiently.
    // A failed integrity check may have partially filled buffers.
    for(auto& field:lj){std::fill(field.values.begin(),field.values.end(),0.);std::fill(field.bad.begin(),field.bad.end(),0);}
    std::fill(phi.values.begin(),phi.values.end(),0.);std::fill(phi.bad.begin(),phi.bad.end(),0);
    std::vector<Query> queries;
    for(auto p:lj_parameters)queries.push_back(make_query(host,p[0],p[1],0.));
    Query coulomb_query(nullptr,poremc_query_destroy);
    if(charged)coulomb_query=make_query(host,0.,0.,1.);
    std::vector<double> reciprocal;
    if(charged&&builder){reciprocal=builder();if(reciprocal.size()!=points)throw std::invalid_argument("host reciprocal grid shape mismatch");}
    constexpr std::size_t chunk=4096;
    std::vector<double> xyz(3*std::min(chunk,points)),values(std::min(chunk,points));
    for(std::size_t begin=0;begin<points;begin+=chunk){
      const std::size_t n=std::min(chunk,points-begin);
      grid_coordinates(begin,n,xyz.data());
      for(std::size_t t=0;t<lj.size();++t){
        if(lj_parameters[t][1]==0.)continue;
        if(poremc_query_energy(queries[t].get(),n,xyz.data(),values.data()))throw std::runtime_error(poremc_last_error());
        for(std::size_t j=0;j<n;++j){const double u=values[j];lj[t].bad[begin+j]=!std::isfinite(u);lj[t].values[begin+j]=std::isfinite(u)?u:0.;}
      }
      if(charged){
        if(poremc_query_energy(coulomb_query.get(),n,xyz.data(),values.data()))throw std::runtime_error(poremc_last_error());
        for(std::size_t j=0;j<n;++j){
          const double u=values[j]+(reciprocal.empty()?0.:reciprocal[begin+j]);
          phi.bad[begin+j]=!std::isfinite(u);phi.values[begin+j]=std::isfinite(u)?u:0.;
        }
      }
    }
  }

  double node(const Type& t,std::size_t k) const {
    const double u=lj[t.lj].values[k]+t.charge*phi.values[k];
    if(cfg.surrogate_cap>0){
      if(lj[t.lj].bad[k]||(t.charge!=0.&&phi.bad[k])||std::isnan(u))return cfg.surrogate_cap;
      return std::clamp(u,-cfg.surrogate_cap,cfg.surrogate_cap);
    }
    return u;
  }

  void pack_stencil_nodes(Type& type) {
    const std::size_t ny=std::size_t(cfg.mesh[1])+3, nz=std::size_t(cfg.mesh[2])+3;
    const std::size_t nx=std::size_t(cfg.mesh[0])+3;
    // An optional in-memory acceleration. Retain the shared raw-field evaluator
    // if a large grid or a highly diverse species would exceed the cache budget.
    constexpr std::size_t budget=192*1024*1024;
    const bool thermal=cfg.steep_interpolation==1 && cfg.surrogate_cap>0 &&
                       cfg.surrogate_cap/cfg.temperature<=500.;
    const std::size_t per_node=(thermal?2:1)*sizeof(double);
    const std::size_t allowed=budget/per_node/types.size();
    if(nx>allowed/ny/nz)return;
    type.packed_energy.resize(nx*ny*nz);
    if(thermal)type.packed_weight.resize(nx*ny*nz);
    for(std::size_t x=0;x<nx;++x)for(std::size_t y=0;y<ny;++y)for(std::size_t z=0;z<nz;++z){
      const auto raw=index(periodic(int(x)-1,0),periodic(int(y)-1,1),periodic(int(z)-1,2));
      const auto here=(x*ny+y)*nz+z;
      const double u=node(type,raw);
      type.packed_energy[here]=u;
      if(thermal)type.packed_weight[here]=std::exp(-u/cfg.temperature);
    }
    statistics.bytes+=per_node*nx*ny*nz;
  }

  void make_stencils() {
    // Mark once. Production avoids discovering the same steep stencil on
    // every trial. A positive-weight interpolation is used around the core.
    const double limit=cfg.surrogate_cap>0?std::min(cfg.steep_limit,.1*cfg.surrogate_cap):cfg.steep_limit;
    for(auto& type:types){
      pack_stencil_nodes(type);
      type.steep.assign(points,0);
      #pragma omp parallel for schedule(static) num_threads(cfg.threads) if(cfg.threads>1)
      for(std::ptrdiff_t c=0;c<static_cast<std::ptrdiff_t>(points);++c){
        const int z=std::size_t(c)%cfg.mesh[2],y=(std::size_t(c)/cfg.mesh[2])%cfg.mesh[1],x=std::size_t(c)/(std::size_t(cfg.mesh[1])*cfg.mesh[2]);
        bool bad=false;double lo=std::numeric_limits<double>::infinity(),hi=-lo;
        for(int a=-1;a<=2&&!bad;++a)for(int b=-1;b<=2&&!bad;++b)for(int d=-1;d<=2;++d){
          const auto k=index(periodic(x+a,0),periodic(y+b,1),periodic(z+d,2));
          const double u=lj[type.lj].values[k]+type.charge*phi.values[k];
          if(lj[type.lj].bad[k]||(type.charge!=0.&&phi.bad[k])||!std::isfinite(u)||std::abs(u)>limit){bad=true;break;}
          lo=std::min(lo,u);hi=std::max(hi,u);
          if(hi-lo>limit){bad=true;break;}
        }
        type.steep[std::size_t(c)]=bad;
      }
      statistics.steep_cells+=std::count(type.steep.begin(),type.steep.end(),std::uint8_t(1));
    }
  }

  double value(std::size_t site,const double* xyz) const {
    const auto& type=types[site_type[site]];++statistics.lookups;
    int base[3];double t[3];
    for(int d=0;d<3;++d){
      double f=xyz[0]*inv[d]+xyz[1]*inv[3+d]+xyz[2]*inv[6+d];f-=std::floor(f);
      const double a=f*cfg.mesh[d];base[d]=std::min(cfg.mesh[d]-1,int(a));t[d]=a-base[d];
    }
    const bool steep=type.steep[index(base[0],base[1],base[2])];
    if(steep&&cfg.surrogate_cap==0.){
      ++statistics.exact_fallbacks;double u;
      if(poremc_query_energy(type.exact.get(),1,xyz,&u))throw std::runtime_error(poremc_last_error());
      if(type.charge!=0.&&reciprocal_exact)u+=reciprocal_exact(1,xyz,&type.charge);
      return u;
    }
    double result=0.;
    if(steep&&cfg.steep_interpolation==1){
      ++statistics.boltzmann_lookups;
      if(!type.packed_weight.empty()){
        // cap/T <= 500 keeps every tabulated weight normal in FP64.
        const std::size_t ny=std::size_t(cfg.mesh[1])+3,nz=std::size_t(cfg.mesh[2])+3;
        const std::size_t lower=((std::size_t(base[0])+1)*ny+base[1]+1)*nz+base[2]+1;
        if(t[0]==0. && t[1]==0. && t[2]==0.)return type.packed_energy[lower];
        double partition=0.;
        for(int a=0;a<2;++a)for(int b=0;b<2;++b){
          const auto* row=type.packed_weight.data()+lower+(a*ny+b)*nz;
          partition+=(a?t[0]:1.-t[0])*(b?t[1]:1.-t[1])*((1.-t[2])*row[0]+t[2]*row[1]);
        }
        return -cfg.temperature*std::log(partition);
      }
      std::array<double,8> weights{},energies{};
      double minimum=std::numeric_limits<double>::infinity();int n=0;
      for(int a=0;a<2;++a)for(int b=0;b<2;++b)for(int c=0;c<2;++c){
        const double weight=(a?t[0]:1-t[0])*(b?t[1]:1-t[1])*(c?t[2]:1-t[2]);
        // Zero-weight corners must not set the shift: an unrelated deep
        // corner could otherwise underflow every nonzero term at a node.
        if(weight<=0.)continue;
        const double energy=node(type,index(periodic(base[0]+a,0),periodic(base[1]+b,1),periodic(base[2]+c,2)));
        weights[n]=weight;energies[n]=energy;++n;minimum=std::min(minimum,energy);
      }
      double partition=0.;
      for(int i=0;i<n;++i)partition+=weights[i]*std::exp(-(energies[i]-minimum)/cfg.temperature);
      result=minimum-cfg.temperature*std::log(partition);
    }else if(steep){
      ++statistics.linear_lookups;
      for(int a=0;a<2;++a)for(int b=0;b<2;++b)for(int c=0;c<2;++c){
        const double weight=(a?t[0]:1-t[0])*(b?t[1]:1-t[1])*(c?t[2]:1-t[2]);
        result+=weight*node(type,index(periodic(base[0]+a,0),periodic(base[1]+b,1),periodic(base[2]+c,2)));
      }
    }else{
      ++statistics.cubic_lookups;
      const auto wx=cubic(t[0]),wy=cubic(t[1]),wz=cubic(t[2]);
      if(!type.packed_energy.empty()){
        const std::size_t ny=std::size_t(cfg.mesh[1])+3,nz=std::size_t(cfg.mesh[2])+3;
        const double* start=type.packed_energy.data()+(std::size_t(base[0])*ny+base[1])*nz+base[2];
        for(int a=0;a<4;++a){
          double uy=0.;
          for(int b=0;b<4;++b){
            const double* row=start+(a*ny+b)*nz;
            uy+=wy[b]*(wz[0]*row[0]+wz[1]*row[1]+wz[2]*row[2]+wz[3]*row[3]);
          }
          result+=wx[a]*uy;
        }
        return result;
      }
      int ix[4],iy[4],iz[4];
      for(int a=0;a<4;++a){ix[a]=periodic(base[0]+a-1,0);iy[a]=periodic(base[1]+a-1,1);iz[a]=periodic(base[2]+a-1,2);}
      for(int a=0;a<4;++a){double uy=0.;for(int b=0;b<4;++b){double uz=0.;
        for(int c=0;c<4;++c)uz+=wz[c]*node(type,index(ix[a],iy[b],iz[c]));
        uy+=wy[b]*uz;}result+=wx[a]*uy;}
    }
    return result;
  }

  bool load() {
    if(cfg.cache_path.empty())return false;
    std::ifstream in(cfg.cache_path,std::ios::binary);if(!in)return false;
    char magic[8];if(!in.read(magic,8)||std::memcmp(magic,"PMCHG061",8))return false;
    std::uint32_t keysize=0;if(!read_scalar(in,keysize)||keysize!=cfg.cache_key.size())return false;
    std::string key(keysize,'\0');if(!in.read(key.data(),key.size())||key!=cfg.cache_key)return false;
    for(int m:cfg.mesh){std::uint32_t stored;if(!read_scalar(in,stored)||stored!=std::uint32_t(m))return false;}
    std::uint64_t nfield=0,npoint=0;
    if(!read_scalar(in,nfield)||!read_scalar(in,npoint)||nfield!=lj.size()||npoint!=points)return false;
    std::array<double,9> cell{};if(!in.read(reinterpret_cast<char*>(cell.data()),sizeof(cell))||cell!=cfg.cell)return false;
    for(auto p:lj_parameters){std::array<double,2> stored{};if(!in.read(reinterpret_cast<char*>(stored.data()),sizeof(stored))||stored!=p)return false;}
    std::uint64_t hash=14695981039346656037ull;
    auto field=[&](Field& f){
      if(!in.read(reinterpret_cast<char*>(f.values.data()),points*sizeof(double)))return false;
      if(!in.read(reinterpret_cast<char*>(f.bad.data()),points*sizeof(std::uint8_t)))return false;
      hash=checksum(f.values.data(),points*sizeof(double),hash);hash=checksum(f.bad.data(),points*sizeof(std::uint8_t),hash);
      return true;
    };
    for(auto& f:lj)if(!field(f))return false;
    if(!field(phi))return false;
    std::uint64_t expected;if(!read_scalar(in,expected)||hash!=expected)return false;
    // The checksum is an integrity check; cache_key is the physics/source
    // identity supplied by Python. No data-dependent sampling state is saved.
    return in.peek()==std::char_traits<char>::eof();
  }

  void save() const {
    if (cfg.cache_key.size() > std::numeric_limits<std::uint32_t>::max())
      throw std::length_error("host grid cache fingerprint exceeds format limit");
    GridCacheWriter out(cfg.cache_path);
    out.write("PMCHG061", 8);
    out.scalar(std::uint32_t(cfg.cache_key.size()));
    out.write(cfg.cache_key.data(), cfg.cache_key.size());
    for (int m : cfg.mesh) out.scalar(std::uint32_t(m));
    out.scalar(std::uint64_t(lj.size()));
    out.scalar(std::uint64_t(points));
    out.write(cfg.cell.data(), cfg.cell.size() * sizeof(double));
    for (const auto& p : lj_parameters) out.write(p.data(), p.size() * sizeof(double));

    std::uint64_t hash = 14695981039346656037ull;
    auto write_field = [&](const Field& field) {
      if (field.values.size() != points || field.bad.size() != points)
        throw std::logic_error("host grid cache field size differs from mesh");
      out.write(field.values.data(), field.values.size() * sizeof(double), &hash);
      out.write(field.bad.data(), field.bad.size() * sizeof(std::uint8_t), &hash);
    };
    for (const auto& field : lj) write_field(field);
    write_field(phi);
    out.scalar(hash);
    out.publish();
  }
};

HostGrid::HostGrid(const HostGridConfig& c,void* env,std::size_t n,const double* s,const double* e,
                   const double* q,HostReciprocalGridBuilder builder,HostReciprocalEnergy exact)
  :p_(std::make_unique<Impl>(c,env,n,s,e,q,std::move(builder),std::move(exact))){}
HostGrid::~HostGrid()=default;
double HostGrid::site_energy(std::size_t site,const double* xyz) const {
  if(site>=p_->site_type.size())throw std::out_of_range("host grid site index");
  return p_->value(site,xyz);
}
void HostGrid::batch_energy(std::size_t batches,std::size_t atoms,const double* xyz,
                            std::size_t first,double* out) const {
  if(first+atoms>p_->site_type.size())throw std::out_of_range("host grid site range");
  for(std::size_t b=0;b<batches;++b){double sum=0.;
    for(std::size_t a=0;a<atoms;++a)sum+=p_->value(first+a,xyz+3*(b*atoms+a));
    out[b]=sum;
  }
}
HostGridStats HostGrid::stats() const{return p_->statistics;}
const HostGridConfig& HostGrid::config() const{return p_->cfg;}
}
