// FP64 cached-neighbour kernels. MIT.
// One block per trial atom; immutable host and query tables stay device-resident.
#include <math.h>
namespace poremc_cuda {
__device__ inline double fractional_component(const double* x,const double* inv,int axis) {
 const double value=x[0]*inv[axis]+x[1]*inv[3+axis]+x[2]*inv[6+axis];
 return value-floor(value);
}

__device__ inline double periodic_r2(double u,double v,double w,const double* h,int diagonal) {
 u-=nearbyint(u);v-=nearbyint(v);w-=nearbyint(w);
 double dx,dy,dz;
 if(diagonal){dx=u*h[0];dy=v*h[4];dz=w*h[8];}
 else{dx=u*h[0]+v*h[3]+w*h[6];dy=u*h[1]+v*h[4]+w*h[7];dz=u*h[2]+v*h[5]+w*h[8];}
 return dx*dx+dy*dy+dz*dz;
}

__device__ inline double pair_value(double r2,double s2,double e4,double qq,
                                    double shift,double force_coefficient,int coulomb,
                                    double rc,double alpha,double vcut,double slope) {
 double value=0;
 if(s2>0&&e4>0) {
  if(r2==0)return INFINITY;
  const double t=s2/r2,t3=t*t*t;
  value=e4*t3*(t3-1)-shift;
  if(force_coefficient!=0.)value-=force_coefficient*(sqrt(r2)-rc)/rc;
 }
 if(coulomb&&qq!=0&&!isinf(value)) {
  if(r2==0)return copysign(INFINITY,qq);
  const double r=sqrt(r2),radial=erfc(alpha*r)/r;
  value+=qq*(coulomb==1?radial-vcut+slope*(r-rc):radial);
 }
 return value;
}
}
extern "C" __global__ void poremc_cached_atoms(
 const long long Q,const int A,const long long M,const double* xyz,
 const double* fs,const double* H,const double* inv,const int* bins,
 const unsigned long long* offsets,const unsigned int* neighbours,
 const double* pair,const int* type_id,const int indexed,const int diagonal,
 const int coulomb,const double rc2,const double rc,const double alpha,
 const double vcut,const double slope,double* atom_out) {
 const long long query=blockIdx.x;if(query>=Q)return;
 const int tid=threadIdx.x,a=int(query%A);const double*x=xyz+3*query;
 const double fx=poremc_cuda::fractional_component(x,inv,0);
 const double fy=poremc_cuda::fractional_component(x,inv,1);
 const double fz=poremc_cuda::fractional_component(x,inv,2);
 unsigned long long begin=0,end=M;
 if(indexed) {
  int ix=int(fx*bins[0]),iy=int(fy*bins[1]),iz=int(fz*bins[2]);
  if(ix>=bins[0])ix=bins[0]-1;if(iy>=bins[1])iy=bins[1]-1;if(iz>=bins[2])iz=bins[2]-1;
  unsigned long long cell=(static_cast<unsigned long long>(ix)*bins[1]+iy)*bins[2]+iz;
  begin=offsets[cell];end=offsets[cell+1];
 }
 const double* table=pair+static_cast<long long>(type_id[a])*4*M;
 double sum=0.0;
 for(unsigned long long k=begin+tid;k<end;k+=blockDim.x) {
  const unsigned long long j=indexed?neighbours[k]:k;
  const double r2=poremc_cuda::periodic_r2(fx-fs[j],fy-fs[M+j],fz-fs[2*M+j],H,diagonal);
  if(r2<rc2)sum+=poremc_cuda::pair_value(r2,table[j],table[M+j],table[3*M+j],
                                      table[2*M+j],0.,coulomb,rc,alpha,vcut,slope);
 }
 __shared__ double values[128];values[tid]=sum;__syncthreads();
 for(int stride=blockDim.x/2;stride>0;stride/=2){if(tid<stride)values[tid]+=values[tid+stride];__syncthreads();}
 if(tid==0)atom_out[query]=values[0];
}
extern "C" __global__ void poremc_sum_atoms(const long long B,const int A,const double* atom_energy,double* out) {
 long long b=static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;if(b>=B)return;
 double value=0;for(int a=0;a<A;++a)value+=atom_energy[b*A+a];out[b]=value;
}
// Optional resident CBMC reduction: only log(mean w), selected index and selected U leave the GPU.
extern "C" __global__ void poremc_select(const long long B,const double* energy,const double* logq,
 const int biased,const double T,const double uniform,double* result) {
 int tid=threadIdx.x;__shared__ double values[128];__shared__ double maximum;
 double local=-INFINITY;
 for(long long b=tid;b<B;b+=blockDim.x){double w=-energy[b]/T-(biased?logq[b]:0);if(w>local)local=w;}
 values[tid]=local;__syncthreads();
 for(int s=blockDim.x/2;s;s/=2){if(tid<s&&values[tid+s]>values[tid])values[tid]=values[tid+s];__syncthreads();}
 if(tid==0)maximum=values[0];__syncthreads();
 if(!isfinite(maximum)){if(tid==0){result[0]=maximum;result[1]=-1;result[2]=INFINITY;}return;}
 double total=0;
 for(long long b=tid;b<B;b+=blockDim.x)total+=exp(-energy[b]/T-(biased?logq[b]:0)-maximum);
 values[tid]=total;__syncthreads();
 for(int s=blockDim.x/2;s;s/=2){if(tid<s)values[tid]+=values[tid+s];__syncthreads();}
 if(tid==0){double cutoff=uniform*values[0],running=0;long long selected=B-1;
  for(long long b=0;b<B;++b){running+=exp(-energy[b]/T-(biased?logq[b]:0)-maximum);if(running>cutoff){selected=b;break;}}
  result[0]=maximum+log(values[0])-log(double(B));result[1]=double(selected);result[2]=energy[selected];}
}
