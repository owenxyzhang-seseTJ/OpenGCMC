#ifndef POREMC_HOST_GRID_INTERP_H
#define POREMC_HOST_GRID_INTERP_H

#include <cmath>
#include <cstddef>

#ifdef __CUDACC__
#define POREMC_GRID_INLINE __host__ __device__ inline
#else
#define POREMC_GRID_INLINE inline
#endif

namespace poremc {

POREMC_GRID_INLINE void cubic_weights(double t,double* weights) {
  // Cardinal cubic with central finite-difference tangents (Catmull-Rom).
  const double t2=t*t,t3=t2*t;
  weights[0]=-.5*t+t2-.5*t3;
  weights[1]=1-2.5*t2+1.5*t3;
  weights[2]=.5*t+2*t2-1.5*t3;
  weights[3]=-.5*t2+.5*t3;
}

// The input has a one-node left halo and two-node right halo on each axis.
// energy and weight point to the start of the 4x4x4 stencil.
POREMC_GRID_INLINE double packed_grid_value(const double* energy,const double* weight,
    std::size_t ny,std::size_t nz,const double* t,bool steep,bool boltzmann,double temperature) {
  if(!steep) {
    double wx[4],wy[4],wz[4],result=0.;
    cubic_weights(t[0],wx);cubic_weights(t[1],wy);cubic_weights(t[2],wz);
    for(int a=0;a<4;++a) {
      double uy=0.;
      for(int b=0;b<4;++b) {
        const auto* row=energy+(a*ny+b)*nz;
        uy+=wy[b]*(wz[0]*row[0]+wz[1]*row[1]+wz[2]*row[2]+wz[3]*row[3]);
      }
      result+=wx[a]*uy;
    }
    return result;
  }

  const std::size_t corner=(ny+1)*nz+1;
  energy+=corner;
  if(boltzmann&&weight) {
    if(t[0]==0.&&t[1]==0.&&t[2]==0.)return energy[0];
    weight+=corner;
    double partition=0.;
    for(int a=0;a<2;++a)for(int b=0;b<2;++b) {
      const auto* row=weight+(a*ny+b)*nz;
      partition+=(a?t[0]:1.-t[0])*(b?t[1]:1.-t[1])*((1.-t[2])*row[0]+t[2]*row[1]);
    }
    return -temperature*::log(partition);
  }

  double result=0.,minimum=INFINITY,energies[8],weights[8];
  int n=0;
  for(int a=0;a<2;++a)for(int b=0;b<2;++b)for(int c=0;c<2;++c) {
    const double w=(a?t[0]:1.-t[0])*(b?t[1]:1.-t[1])*(c?t[2]:1.-t[2]);
    const double u=energy[(a*ny+b)*nz+c];
    if(!boltzmann)result+=w*u;
    else if(w>0.) {
      // A zero-weight corner cannot determine the shift at an exact node.
      energies[n]=u;weights[n]=w;++n;
      if(u<minimum)minimum=u;
    }
  }
  if(!boltzmann)return result;
  double partition=0.;
  for(int i=0;i<n;++i)partition+=weights[i]*::exp(-(energies[i]-minimum)/temperature);
  return minimum-temperature*::log(partition);
}

}
#undef POREMC_GRID_INLINE
#endif
