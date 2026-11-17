// Paired direct/cached CUDA benchmark. Requires nvcc and a device; no CuPy dependency.
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "../poremc/energies_cuda.cu"
#include "../poremc/cached_cuda.cu"
static void check(cudaError_t s){if(s!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(s));}
template<class T>T read_one(std::ifstream&f){T v;f.read(reinterpret_cast<char*>(&v),sizeof(v));if(!f)throw std::runtime_error("truncated input");return v;}
template<class T>std::vector<T> read_vec(std::ifstream&f,size_t n){std::vector<T>v(n);if(n)f.read(reinterpret_cast<char*>(v.data()),n*sizeof(T));if(!f)throw std::runtime_error("truncated array");return v;}
template<class T>struct Device{T*p=nullptr;Device(size_t n){check(cudaMalloc(reinterpret_cast<void**>(&p),std::max(size_t(1),n)*sizeof(T)));}~Device(){if(p)cudaFree(p);}Device(const Device&)=delete;Device&operator=(const Device&)=delete;void upload(const std::vector<T>&v){if(!v.empty())check(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}};
static void write_values(const std::string&p,const std::vector<double>&x){std::ofstream f(p,std::ios::binary);if(!x.empty())f.write(reinterpret_cast<const char*>(x.data()),x.size()*8);if(!f)throw std::runtime_error("cannot write outputs");}
static void json_vector(std::ofstream&f,const std::vector<double>&x){f<<'[';for(size_t i=0;i<x.size();++i){if(i)f<<',';f<<x[i];}f<<']';}
int main(int argc,char**argv){try{
 if(argc!=5)throw std::runtime_error("usage: gpu_cached_driver fixture.bin plan.bin output_prefix repeats");
 const int repeats=std::stoi(argv[4]);if(repeats<0||repeats>10000)throw std::runtime_error("invalid repeats");
 std::ifstream f(argv[1],std::ios::binary);auto B=read_one<long long>(f),A=read_one<long long>(f),M=read_one<long long>(f);
 double rc=read_one<double>(f),alpha=read_one<double>(f);int shifted=read_one<int>(f),mode=read_one<int>(f);
 if(B<0||A<0||M<0||A>INT32_MAX)throw std::runtime_error("invalid fixture dimensions");
 auto old=read_vec<double>(f,3*B*A+3*A+6*M+18);
 std::ifstream c(argv[2],std::ios::binary);char magic[8];c.read(magic,8);if(std::memcmp(magic,"PCACHE02",8))throw std::runtime_error("wrong cache schema");
 auto bins=read_vec<int>(c,3);int indexed=read_one<int>(c),diagonal=read_one<int>(c),types=read_one<int>(c);
 auto no=read_one<std::uint64_t>(c),ni=read_one<std::uint64_t>(c);
 auto fs=read_vec<double>(c,3*M);auto pair=read_vec<double>(c,size_t(types)*4*M);auto ids=read_vec<int>(c,A);
 auto offsets=read_vec<unsigned long long>(c,no);auto neighbours=read_vec<unsigned int>(c,ni);
 check(cudaSetDevice(0));cudaDeviceProp property;check(cudaGetDeviceProperties(&property,0));
 int driver=0,runtime=0;check(cudaDriverGetVersion(&driver));check(cudaRuntimeGetVersion(&runtime));
 auto setup_start=std::chrono::steady_clock::now();
 Device<double>dold(old.size()),dfs(fs.size()),dpair(pair.size()),partial(B*A),output(B);
 Device<int>dbins(3),dids(ids.size());Device<unsigned long long>doff(offsets.size());Device<unsigned int>dnei(neighbours.size());
 dold.upload(old);dfs.upload(fs);dpair.upload(pair);dbins.upload(bins);dids.upload(ids);doff.upload(offsets);dnei.upload(neighbours);
 const double *xyz=dold.p,*ts=xyz+3*B*A,*te=ts+A,*tq=te+A,*oldfs=tq+A,*hs=oldfs+3*M,*he=hs+M,*hq=he+M,*H=hq+M,*inv=H+9;
 check(cudaDeviceSynchronize());double setup_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-setup_start).count();
 const double vcut=std::erfc(alpha*rc)/rc,slope=vcut/rc+2*alpha/std::sqrt(3.14159265358979323846)*std::exp(-alpha*alpha*rc*rc)/rc;
 auto launch=[&](bool cached){
  if(!cached){if(B)poremc_trials<<<static_cast<unsigned>(B),128,128*sizeof(double)>>>(B,A,M,xyz,ts,te,tq,oldfs,hs,he,hq,H,inv,rc,alpha,shifted,mode,output.p);}
  else{if(B*A)poremc_cached_atoms<<<static_cast<unsigned>(B*A),128>>>(B*A,int(A),M,xyz,dfs.p,H,inv,dbins.p,doff.p,dnei.p,dpair.p,dids.p,indexed,diagonal,mode,rc*rc,rc,alpha,vcut,slope,partial.p);
       if(B)poremc_sum_atoms<<<static_cast<unsigned>((B+127)/128),128>>>(B,int(A),partial.p,output.p);}
  check(cudaGetLastError());
 };
 std::vector<double>values(B);std::string prefix=argv[3];
 for(bool cached:{false,true}){launch(cached);check(cudaDeviceSynchronize());if(B)check(cudaMemcpy(values.data(),output.p,B*8,cudaMemcpyDeviceToHost));write_values(prefix+(cached?"_cached.bin":"_direct.bin"),values);}
 std::vector<double>kernels[2],calls[2];cudaEvent_t begin,end;check(cudaEventCreate(&begin));check(cudaEventCreate(&end));
 for(int repeat=0;repeat<repeats;++repeat){
  // Alternate order to reduce monotone temperature/order effects.
  for(int j=0;j<2;++j){int k=(repeat+j)%2;check(cudaEventRecord(begin));launch(k);check(cudaEventRecord(end));check(cudaEventSynchronize(end));float ms=0;check(cudaEventElapsedTime(&ms,begin,end));kernels[k].push_back(ms);}
  for(int j=0;j<2;++j){int k=(repeat+j)%2;auto start=std::chrono::steady_clock::now();if(B*A)check(cudaMemcpy(dold.p,old.data(),B*A*3*8,cudaMemcpyHostToDevice));launch(k);if(B)check(cudaMemcpy(values.data(),output.p,B*8,cudaMemcpyDeviceToHost));else check(cudaDeviceSynchronize());calls[k].push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());}
 }
 check(cudaEventDestroy(begin));check(cudaEventDestroy(end));
 std::ofstream out(prefix+".json");out<<std::setprecision(17)<<"{\"device\":\""<<property.name<<"\",\"compute_major\":"<<property.major<<",\"compute_minor\":"<<property.minor<<",\"driver\":"<<driver<<",\"runtime\":"<<runtime<<",\"setup_ms\":"<<setup_ms<<",\"indexed\":"<<indexed<<",\"direct_kernel_ms\":";json_vector(out,kernels[0]);out<<",\"cached_kernel_ms\":";json_vector(out,kernels[1]);out<<",\"direct_coordinate_transfer_call_ms\":";json_vector(out,calls[0]);out<<",\"cached_coordinate_transfer_call_ms\":";json_vector(out,calls[1]);out<<"}\n";
 return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
