// PoreMC 0.2, independent FP64 cached-neighbour kernels. MIT.
// One block per trial atom; immutable host and query tables stay device-resident.
#include <math.h>
extern "C" __global__ void poremc_cached_atoms(
 const long long Q,const int A,const long long M,const double* xyz,
 const double* fs,const double* H,const double* inv,const int* bins,
 const unsigned long long* offsets,const unsigned int* neighbours,
 const double* pair,const int* type_id,const int indexed,const int diagonal,
 const int coulomb,const double rc2,const double rc,const double alpha,
 const double vcut,const double slope,double* atom_out) {
 const long long query=blockIdx.x;if(query>=Q)return;
 const int tid=threadIdx.x,a=int(query%A);const double*x=xyz+3*query;
 double fx=x[0]*inv[0]+x[1]*inv[3]+x[2]*inv[6];fx-=floor(fx);
 double fy=x[0]*inv[1]+x[1]*inv[4]+x[2]*inv[7];fy-=floor(fy);
 double fz=x[0]*inv[2]+x[1]*inv[5]+x[2]*inv[8];fz-=floor(fz);
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
  double u=fx-fs[j],v=fy-fs[M+j],w=fz-fs[2*M+j];
  u-=nearbyint(u);v-=nearbyint(v);w-=nearbyint(w);
  double dx,dy,dz;
  if(diagonal){dx=u*H[0];dy=v*H[4];dz=w*H[8];}
  else{dx=u*H[0]+v*H[3]+w*H[6];dy=u*H[1]+v*H[4]+w*H[7];dz=u*H[2]+v*H[5]+w*H[8];}
  const double r2=dx*dx+dy*dy+dz*dz;
  if(r2<rc2) {
   const double s2=table[j],e4=table[M+j],qq=table[3*M+j];double value=0;
   if(s2>0&&e4>0) {
    if(r2==0)value=INFINITY;
    else{double t=s2/r2,t3=t*t*t;value=e4*t3*(t3-1)-table[2*M+j];}
   }
   if(coulomb&&qq!=0&&!isinf(value)) {
    if(r2==0)value=copysign(INFINITY,qq);
    else{double r=sqrt(r2),radial=erfc(alpha*r)/r;value+=qq*(coulomb==1?radial-vcut+slope*(r-rc):radial);}
   }
   sum+=value;
  }
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
