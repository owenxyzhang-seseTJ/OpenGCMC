// Independent FP64 CUDA direct-pair trial reduction. Requires CuPy/CUDA.
// Direct traversal reference kernel for paired FP64 energy checks.

#include <math.h>

extern "C" __global__ void poremc_trials(
 const long long B,const long long A,const long long M,
 const double* xyz,const double* ts,const double* te,const double* tq,
 const double* fs,const double* hs,const double* he,const double* hq,
 const double* H,const double* inv,const double rc,const double alpha,
 const int shifted,const int coulomb,double* out) {
 const long long b=blockIdx.x;
 const int tid=threadIdx.x;
 const double ck=167100.94689828736;
 const double vcut=erfc(alpha*rc)/rc;
 const double slope=erfc(alpha*rc)/(rc*rc)+1.1283791670955125739*alpha*exp(-alpha*alpha*rc*rc)/rc;
 double sum=0.0;
 for(long long p=tid;p<A*M;p+=blockDim.x) {
   const long long a=p/M,j=p%M;
   const double* x=xyz+(b*A+a)*3;
   double fx=x[0]*inv[0]+x[1]*inv[3]+x[2]*inv[6]-fs[j*3];
   double fy=x[0]*inv[1]+x[1]*inv[4]+x[2]*inv[7]-fs[j*3+1];
   double fz=x[0]*inv[2]+x[1]*inv[5]+x[2]*inv[8]-fs[j*3+2];
   fx-=nearbyint(fx);fy-=nearbyint(fy);fz-=nearbyint(fz);
   const double dx=fx*H[0]+fy*H[3]+fz*H[6];
   const double dy=fx*H[1]+fy*H[4]+fz*H[7];
   const double dz=fx*H[2]+fy*H[5]+fz*H[8];
   const double r2=dx*dx+dy*dy+dz*dz;
   if(r2<rc*rc) {
     const double s=0.5*(ts[a]+hs[j]),e=sqrt(te[a]*he[j]);
     double value=0.0;
     if(s>0 && e>0) {
       if(r2==0) value=INFINITY;
       else {
         const double u=s*s/r2,u3=u*u*u;
         value=4*e*u3*(u3-1);
         if(shifted) {const double c=s*s/(rc*rc),c3=c*c*c;value-=4*e*c3*(c3-1);}
       }
     }
     const double qq=tq[a]*hq[j];
     if(coulomb && qq!=0 && !isinf(value)) {
       if(r2==0) value=copysign(INFINITY,qq);
       else {const double r=sqrt(r2);
         const double radial=erfc(alpha*r)/r;
         value+=ck*qq*(coulomb==1 ? radial-vcut+slope*(r-rc) : radial);}
     }
     sum+=value;
   }
 }
 extern __shared__ double scratch[];
 scratch[tid]=sum;__syncthreads();
 for(unsigned stride=blockDim.x/2;stride>0;stride/=2) {
   if(tid<stride) scratch[tid]+=scratch[tid+stride];
   __syncthreads();
 }
 if(tid==0) out[b]=scratch[0];
}
