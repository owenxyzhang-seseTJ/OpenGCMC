// Optional native CUDA plug-in for simultaneous CBMC reciprocal candidates.
// Build with nvcc -O3 --fmad=false -std=c++17 -Xcompiler -fPIC -shared \
//   native/ewald_cuda.cu -o poremc/libporemc_ewald_cuda.so
// No Python callback is used by the native FastGCMC sampling loop.
#include "ewald_cuda.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
thread_local std::string last_error;
void check(cudaError_t status,const char*operation){if(status!=cudaSuccess)throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(status));}
struct Buffer {
 double* data=nullptr;size_t capacity=0;
 ~Buffer(){if(data)cudaFree(data);}
 void reserve(size_t count){if(count<=capacity)return;double*next=nullptr;check(cudaMalloc(reinterpret_cast<void**>(&next),count*sizeof(double)),"cudaMalloc");if(data)cudaFree(data);data=next;capacity=count;}
};
struct GPU {
 int device=0;size_t modes=0;cudaStream_t stream=nullptr;
 Buffer k,weight,xyz,q,base,delta,partial,result;
 std::vector<double> previous_base;
 GPU(size_t n,const double*kv,const double*w,int d):device(d),modes(n){
  check(cudaSetDevice(device),"cudaSetDevice");check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"cudaStreamCreate");
  try{k.reserve(3*n);weight.reserve(n);base.reserve(2*n);check(cudaMemcpyAsync(k.data,kv,3*n*sizeof(double),cudaMemcpyHostToDevice,stream),"upload k vectors");check(cudaMemcpyAsync(weight.data,w,n*sizeof(double),cudaMemcpyHostToDevice,stream),"upload reciprocal weights");check(cudaStreamSynchronize(stream),"initialize reciprocal state");}
  catch(...){cudaStreamDestroy(stream);stream=nullptr;throw;}
 }
 ~GPU(){cudaSetDevice(device);if(stream)cudaStreamDestroy(stream);}
};
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
}
extern "C" {
void* poremc_ewald_cuda_create(size_t modes,const double*k,const double*weights,int device){
 try{last_error.clear();if(!modes)throw std::invalid_argument("CUDA Ewald requires reciprocal modes");return new GPU(modes,k,weights,device);}catch(const std::exception&e){last_error=e.what();return nullptr;}
}
void poremc_ewald_cuda_destroy(void*p){delete static_cast<GPU*>(p);}
const char* poremc_ewald_cuda_last_error(){return last_error.c_str();}
int poremc_ewald_cuda_batch(void*p,size_t batches,size_t atoms,const double*xyz,const double*q,const double*base,double base_charge,double self_coefficient,double background_coefficient,double*out){
 try{
  last_error.clear();if(!p)throw std::invalid_argument("null CUDA reciprocal state");if(!batches)return 0;
  GPU&g=*static_cast<GPU*>(p);check(cudaSetDevice(g.device),"cudaSetDevice");
  const size_t terms=batches*g.modes;
  g.xyz.reserve(3*batches*atoms);g.q.reserve(atoms);g.delta.reserve(2*terms);g.partial.reserve(terms);g.result.reserve(batches);
  check(cudaMemcpyAsync(g.xyz.data,xyz,3*batches*atoms*sizeof(double),cudaMemcpyHostToDevice,g.stream),"upload candidates");
  check(cudaMemcpyAsync(g.q.data,q,atoms*sizeof(double),cudaMemcpyHostToDevice,g.stream),"upload trial charges");
  if(g.previous_base.size()!=2*g.modes||!std::equal(g.previous_base.begin(),g.previous_base.end(),base)){
   check(cudaMemcpyAsync(g.base.data,base,2*g.modes*sizeof(double),cudaMemcpyHostToDevice,g.stream),"upload base structure factor");g.previous_base.assign(base,base+2*g.modes);
  }
  double charge=0,squared=0;for(size_t a=0;a<atoms;++a){charge+=q[a];squared+=q[a]*q[a];}
  const double correction=self_coefficient*squared+background_coefficient*(2*base_charge*charge+charge*charge);
  candidates_kernel<<<unsigned((terms+255)/256),256,0,g.stream>>>(g.modes,atoms,batches,g.k.data,g.weight.data,g.xyz.data,g.q.data,g.base.data,g.delta.data,g.partial.data);
  check(cudaPeekAtLastError(),"launch candidate structure factors");
  reduce_kernel<<<unsigned(batches),256,0,g.stream>>>(g.modes,g.partial.data,correction,g.result.data);check(cudaPeekAtLastError(),"launch candidate reduction");
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
