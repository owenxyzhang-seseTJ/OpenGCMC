// Optional native CUDA plug-in for simultaneous CBMC reciprocal candidates.
// Build with nvcc -O3 --fmad=false -std=c++17 -Xcompiler -fPIC -shared \
//   native/ewald_cuda.cu -o poremc/libporemc_ewald_cuda.so
// No Python callback is used by the native FastGCMC sampling loop.
#include "ewald_cuda.h"
#include "cuda_buffers.h"
#include "ewald_cuda_tensor.h"
#include <limits>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
using namespace poremc_cuda_tensor;
thread_local std::string last_error;
using poremc_cuda::Buffer;
using poremc_cuda::check;
struct GPU {
 int device=0;size_t modes=0;cudaStream_t stream=nullptr;
 Buffer<double> k,weight,xyz,q,base,delta,partial,result;
 Buffer<Complex> powers,selected,resident;
 Buffer<Tile> tiles;
 Buffer<int> z_index;
 Axes axes{};
 size_t tile_count=0;
 std::string device_name;
 std::vector<double> previous_base;
 GPU(size_t n,const double*kv,const double*w,int d):device(d),modes(n){
  check(cudaSetDevice(device),"cudaSetDevice");cudaDeviceProp property{};check(cudaGetDeviceProperties(&property,device),"query CUDA device");device_name=property.name;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"cudaStreamCreate");
  try{k.reserve(3*n);weight.reserve(n);base.reserve(2*n);check(cudaMemcpyAsync(k.data,kv,3*n*sizeof(double),cudaMemcpyHostToDevice,stream),"upload k vectors");check(cudaMemcpyAsync(weight.data,w,n*sizeof(double),cudaMemcpyHostToDevice,stream),"upload reciprocal weights");check(cudaStreamSynchronize(stream),"initialize reciprocal state");}
  catch(...){cudaStreamDestroy(stream);stream=nullptr;throw;}
 }
 ~GPU(){cudaSetDevice(device);if(stream)cudaStreamDestroy(stream);}
};
__global__ void axis_kernel(Axes axes,size_t sites,const double* xyz,Complex* powers) {
 for(size_t u=size_t(blockIdx.x)*blockDim.x+threadIdx.x;u<sites*3;u+=size_t(gridDim.x)*blockDim.x){
  const size_t site=u/3;
  harmonics(axes,int(u%3),xyz+3*site,powers+site*axes.span);
 }
}
// A warp owns up to 32 consecutive retained modes with one (kx,ky). Its lanes
// first form U[b,xy,a]=q[a]*X[b,a,x]*Y[b,a,y], then broadcast U as they
// contract S[b,xy,z]=sum_a U[b,xy,a]*Z[b,a,z]. No [B,K,A] array is stored.
// Keep every lane participating in shuffles, including padded ends of rows.
template<bool ExportStructure>
__global__ void contraction_kernel(size_t batches,size_t atoms,size_t tile_count,int span,
 const Tile* tiles,const int* z_index,const Complex* powers,const double* charge,
 const double* weight,const double* base,double* tile_energy,Complex* selected) {
 constexpr unsigned mask=0xffffffffu;
 const int lane=threadIdx.x&31;
 for(size_t u=(size_t(blockIdx.x)*blockDim.x+threadIdx.x)/32;u<batches*tile_count;
     u+=size_t(gridDim.x)*(blockDim.x/32)) {
  const size_t b=u/tile_count;
  const Tile tile=tiles[u%tile_count];
  const bool active=lane<tile.count;
  const int mode=active ? tile.first+lane : tile.first;
  const int iz=z_index[mode];
  const Complex* molecule=powers+b*atoms*span;
  Complex s{0.0,0.0};
  for(size_t first=0;first<atoms;first+=32){
   const size_t atom=first+lane;
   Complex xy{0.0,0.0};
   if(atom<atoms){
    const Complex* v=molecule+atom*span;
    xy=multiply(v[tile.ix],v[tile.iy]);
    xy.re*=charge[atom];xy.im*=charge[atom];
   }
   const int count=atoms-first<32 ? static_cast<int>(atoms-first) : 32;
   for(int a=0;a<count;++a){
    const Complex ua{__shfl_sync(mask,xy.re,a),__shfl_sync(mask,xy.im,a)};
    const Complex product=multiply(ua,molecule[(first+a)*span+iz]);
    s.re+=product.re;s.im+=product.im;
   }
  }
  if constexpr (ExportStructure){
   if(active)selected[mode]=s;  // only used with B=1
  } else {
   double local=active ? energy(weight[mode],base[2*mode],base[2*mode+1],s) : 0.0;
   for(int step=16;step;step/=2)local+=__shfl_down_sync(mask,local,step);
   if(lane==0)tile_energy[u]=local;
  }
 }
}
size_t checked_product(size_t a,size_t b){
 if(a && b>std::numeric_limits<size_t>::max()/a)throw std::overflow_error("CUDA candidate dimensions overflow");
 return a*b;
}
unsigned blocks_for(size_t tasks,unsigned per_block){
 return static_cast<unsigned>(std::min<size_t>(1+(tasks-1)/per_block,65535));
}
void upload_candidates(GPU& g,size_t b,size_t a,const double* xyz,const double* q){
 const size_t coordinates=checked_product(3,checked_product(b,a));
 g.xyz.reserve(coordinates);g.q.reserve(a);
 check(cudaMemcpyAsync(g.xyz.data,xyz,coordinates*sizeof(double),cudaMemcpyHostToDevice,g.stream),"upload candidates");
 check(cudaMemcpyAsync(g.q.data,q,a*sizeof(double),cudaMemcpyHostToDevice,g.stream),"upload charges");
}
void tensor_phases(GPU& g,size_t b,size_t a){
 const size_t sites=checked_product(b,a);
 g.powers.reserve(checked_product(sites,size_t(g.axes.span)));
 axis_kernel<<<blocks_for(checked_product(sites,3),128),128,0,g.stream>>>(g.axes,sites,g.xyz.data,g.powers.data);
 check(cudaPeekAtLastError(),"launch reciprocal harmonics");
}

__global__ void candidates_kernel(size_t modes,size_t atoms,size_t batches,
 const double* k,const double*weights,const double*xyz,const double*q,
 const double*base,double*delta,double*partial){
 const size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
 if(index>=batches*modes)return;
 const size_t b=index/modes,mode=index-b*modes;
 double re=0,im=0;
 for(size_t a=0;a<atoms;++a){double qa=q[a];if(qa==0)continue;size_t pos=3*(b*atoms+a);double angle=k[3*mode]*xyz[pos]+k[3*mode+1]*xyz[pos+1]+k[3*mode+2]*xyz[pos+2];double sine,cosine;sincos(angle,&sine,&cosine);re+=qa*cosine;im+=qa*sine;}
 // Explicit delta S is retained on-device for every [candidate, reciprocal mode].
 delta[2*index]=re;delta[2*index+1]=im;
 partial[index]=weights[mode]*(2*(base[2*mode]*re+base[2*mode+1]*im)+re*re+im*im);
}
__global__ void reduce_kernel(size_t modes,const double*partial,double correction,double*out){
 __shared__ double sums[256];const size_t b=blockIdx.x;double sum=0;
 for(size_t i=threadIdx.x;i<modes;i+=blockDim.x)sum+=partial[b*modes+i];sums[threadIdx.x]=sum;__syncthreads();
 for(unsigned width=blockDim.x/2;width; width>>=1){if(threadIdx.x<width)sums[threadIdx.x]+=sums[threadIdx.x+width];__syncthreads();}
 if(threadIdx.x==0)out[b]=sums[0]+correction;
}
template<bool Commit>
__global__ void resident_delta(size_t modes,Complex* bath,const Complex* change,const double* weights,double* out){
 for(size_t k=size_t(blockIdx.x)*blockDim.x+threadIdx.x;k<modes;k+=size_t(gridDim.x)*blockDim.x){
  const Complex d=change[k];
  if constexpr (Commit){
   bath[k].re+=d.re;
   bath[k].im+=d.im;
  }else out[k]=energy(weights[k],bath[k].re,bath[k].im,d);
 }
}
}
extern "C" {
void* poremc_ewald_cuda_create(size_t modes,const double*k,const double*weights,int device){
 try{last_error.clear();if(!modes)throw std::invalid_argument("CUDA Ewald requires reciprocal modes");return new GPU(modes,k,weights,device);}catch(const std::exception&e){last_error=e.what();return nullptr;}
}
int poremc_ewald_cuda_set_lattice(void* p,const double* inverse,const int* limits,const int* indices){
 try {
  last_error.clear();if(!p)throw std::invalid_argument("null CUDA state");
  GPU& g=*static_cast<GPU*>(p);
  const auto lattice=plan(g.modes,inverse,limits,indices);
  check(cudaSetDevice(g.device),"select CUDA lattice device");
  // Synchronous initialization ensures the local immutable plan can be released even on error.
  g.tile_count=0;
  g.tiles.reserve(lattice.tiles.size());g.z_index.reserve(g.modes);
  check(cudaMemcpy(g.tiles.data,lattice.tiles.data(),lattice.tiles.size()*sizeof(Tile),cudaMemcpyHostToDevice),"upload CUDA mode tiles");
  check(cudaMemcpy(g.z_index.data,lattice.z_index.data(),g.modes*sizeof(int),cudaMemcpyHostToDevice),"upload CUDA z indices");
  g.axes=lattice.axes;g.tile_count=lattice.tiles.size();return 0;
 }catch(const std::exception& e){last_error=e.what();return -1;}
}
const char* poremc_ewald_cuda_kernel_name(void* p){return p && static_cast<GPU*>(p)->tile_count ? "cuda-tensor" : "cuda";}
const char* poremc_ewald_cuda_device_name(void* p){return p ? static_cast<GPU*>(p)->device_name.c_str() : "";}
int poremc_ewald_cuda_resident_import(void* p,const double* modes){
 try{
  GPU& g=*static_cast<GPU*>(p);check(cudaSetDevice(g.device),"select CUDA resident device");
  g.resident.reserve(g.modes);
  static_assert(sizeof(Complex)==2*sizeof(double),"unexpected complex layout");
  check(cudaMemcpyAsync(g.resident.data,modes,g.modes*sizeof(Complex),cudaMemcpyHostToDevice,g.stream),"upload resident spectrum");
  check(cudaStreamSynchronize(g.stream),"finish spectrum upload");
  return 0;
 }catch(const std::exception& e){last_error=e.what();return -1;}
}
int poremc_ewald_cuda_resident_export(void* p,double* modes){
 try{
  GPU& g=*static_cast<GPU*>(p);check(cudaSetDevice(g.device),"select CUDA resident device");
  check(cudaMemcpyAsync(modes,g.resident.data,g.modes*sizeof(Complex),cudaMemcpyDeviceToHost,g.stream),"read resident spectrum");
  check(cudaStreamSynchronize(g.stream),"finish spectrum read");
  return 0;
 }catch(const std::exception& e){last_error=e.what();return -1;}
}
int poremc_ewald_cuda_resident_trial(void* p,size_t atoms,const double* xyz,const double* q,double* out){
 auto& g=*static_cast<GPU*>(p);
 try{
  last_error.clear();
  if(!g.resident.data||!g.tile_count||!atoms)throw std::invalid_argument("invalid resident Ewald trial");
  check(cudaSetDevice(g.device),"select CUDA resident device");
  g.selected.reserve(g.modes);g.partial.reserve(g.modes);g.result.reserve(1);
  upload_candidates(g,1,atoms,xyz,q);tensor_phases(g,1,atoms);
  contraction_kernel<true><<<blocks_for(g.tile_count,4),128,0,g.stream>>>(
      1,atoms,g.tile_count,g.axes.span,g.tiles.data,g.z_index.data,g.powers.data,g.q.data,
      nullptr,nullptr,nullptr,g.selected.data);
  check(cudaPeekAtLastError(),"launch resident structure-factor increment");
  resident_delta<false><<<blocks_for(g.modes,256),256,0,g.stream>>>(
      g.modes,g.resident.data,g.selected.data,g.weight.data,g.partial.data);
  check(cudaPeekAtLastError(),"launch resident energy increment");
  reduce_kernel<<<1,256,0,g.stream>>>(g.modes,g.partial.data,0.,g.result.data);
  check(cudaPeekAtLastError(),"reduce resident trial energy");
  check(cudaMemcpyAsync(out,g.result.data,sizeof(double),cudaMemcpyDeviceToHost,g.stream),"read resident trial energy");
  check(cudaStreamSynchronize(g.stream),"finish resident trial");
  return 0;
 }catch(const std::exception& e){cudaSetDevice(g.device);cudaStreamSynchronize(g.stream);last_error=e.what();return -1;}
}
int poremc_ewald_cuda_resident_commit(void* p){
 try{
  GPU& g=*static_cast<GPU*>(p);
  check(cudaSetDevice(g.device),"select CUDA resident device");
  resident_delta<true><<<blocks_for(g.modes,256),256,0,g.stream>>>(
      g.modes,g.resident.data,g.selected.data,nullptr,nullptr);
  check(cudaPeekAtLastError(),"commit resident structure-factor increment");
  return 0;
 }catch(const std::exception& e){last_error=e.what();return -1;}
}
int poremc_ewald_cuda_structure(void* p,size_t atoms,const double* xyz,const double* q,double* out){
 try{
  last_error.clear();if(!p||!out)throw std::invalid_argument("missing CUDA state/structure output");
  GPU& g=*static_cast<GPU*>(p);
  if(!g.tile_count)throw std::invalid_argument("selected structure requires CUDA tensor mode tiles");
  if(!atoms){std::fill_n(out,2*g.modes,0.0);return 0;}
  check(cudaSetDevice(g.device),"select CUDA structure device");
  upload_candidates(g,1,atoms,xyz,q);tensor_phases(g,1,atoms);
  g.selected.reserve(g.modes);
  contraction_kernel<true><<<blocks_for(g.tile_count,4),128,0,g.stream>>>(1,atoms,g.tile_count,g.axes.span,g.tiles.data,g.z_index.data,g.powers.data,g.q.data,nullptr,nullptr,nullptr,g.selected.data);
  check(cudaPeekAtLastError(),"launch selected tensor structure");
  static_assert(sizeof(Complex)==2*sizeof(double),"unexpected complex layout");
  check(cudaMemcpyAsync(out,g.selected.data,g.modes*sizeof(Complex),cudaMemcpyDeviceToHost,g.stream),"download selected structure");
  check(cudaStreamSynchronize(g.stream),"finish selected structure");return 0;
 }catch(const std::exception& e){
  if(p){auto& g=*static_cast<GPU*>(p);cudaSetDevice(g.device);cudaStreamSynchronize(g.stream);}
  last_error=e.what();return -1;
 }
}
void poremc_ewald_cuda_destroy(void*p){delete static_cast<GPU*>(p);}
const char* poremc_ewald_cuda_last_error(){return last_error.c_str();}
int poremc_ewald_cuda_batch(void*p,size_t batches,size_t atoms,const double*xyz,const double*q,const double*base,double base_charge,double self_coefficient,double background_coefficient,double*out){
 try{
  last_error.clear();if(!p)throw std::invalid_argument("null CUDA reciprocal state");if(!batches)return 0;
  GPU&g=*static_cast<GPU*>(p);check(cudaSetDevice(g.device),"cudaSetDevice");
  if(!atoms){std::fill_n(out,batches,0.0);return 0;}
  const size_t terms=checked_product(batches,g.modes);
  if(batches>2147483647u)throw std::invalid_argument("CUDA batch exceeds grid dimension");
  g.result.reserve(batches);
  upload_candidates(g,batches,atoms,xyz,q);
  if(g.previous_base.size()!=2*g.modes||!std::equal(g.previous_base.begin(),g.previous_base.end(),base)){
   check(cudaMemcpyAsync(g.base.data,base,2*g.modes*sizeof(double),cudaMemcpyHostToDevice,g.stream),"upload base structure factor");g.previous_base.assign(base,base+2*g.modes);
  }
  double charge=0,squared=0;for(size_t a=0;a<atoms;++a){charge+=q[a];squared+=q[a]*q[a];}
  const double correction=self_coefficient*squared+background_coefficient*(2*base_charge*charge+charge*charge);
  size_t stride=g.modes;
  if(g.tile_count){
   const size_t tiles=checked_product(g.tile_count,batches);
   g.partial.reserve(tiles);tensor_phases(g,batches,atoms);
   contraction_kernel<false><<<blocks_for(tiles,4),128,0,g.stream>>>(batches,atoms,g.tile_count,g.axes.span,g.tiles.data,g.z_index.data,g.powers.data,g.q.data,g.weight.data,g.base.data,g.partial.data,nullptr);
   stride=g.tile_count;
  }else{
   g.delta.reserve(checked_product(2,terms));g.partial.reserve(terms);
   if(1+(terms-1)/256>2147483647u)throw std::invalid_argument("CUDA direct batch exceeds grid dimension");
   candidates_kernel<<<static_cast<unsigned>(1+(terms-1)/256),256,0,g.stream>>>(g.modes,atoms,batches,g.k.data,g.weight.data,g.xyz.data,g.q.data,g.base.data,g.delta.data,g.partial.data);
  }
  check(cudaPeekAtLastError(),"launch candidate structure factors");
  reduce_kernel<<<unsigned(batches),256,0,g.stream>>>(stride,g.partial.data,correction,g.result.data);check(cudaPeekAtLastError(),"launch candidate reduction");
  check(cudaMemcpyAsync(out,g.result.data,batches*sizeof(double),cudaMemcpyDeviceToHost,g.stream),"download candidate energies");
  check(cudaStreamSynchronize(g.stream),"finish candidate batch");return 0;
 }catch(const std::exception&e){
  // Complete queued transfers before the caller's temporary host arrays die.
  // A failed batch never validates the retained base-SF upload cache.
  if(p){GPU&g=*static_cast<GPU*>(p);cudaSetDevice(g.device);if(g.stream)cudaStreamSynchronize(g.stream);g.previous_base.clear();}
  last_error=e.what();return -1;
 }
}
}
