// Periodic skew cell, signed host potential and near-wall steep cells.
// Compare identical coordinates and exported energies in two release binaries.
#include "host_grid.h"
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>
extern "C" void* poremc_environment_create(std::size_t,const double*,const double*,const double*,const double*,const double*,double,int,double,int,int,int);
extern "C" void poremc_environment_destroy(void*);
int main(int argc,char**argv) {
  const int m=64;
  const double T=argc>1?std::stod(argv[1]):350.;
  std::array<double,9> cell={38.,0.,0.,7.,41.,0.,-3.,5.,43.};
  const double zero[3]={0,0,0}, q[3]={1.,-.25,.5};
  auto* host=poremc_environment_create(0,zero,zero,zero,zero,cell.data(),8.,2,.28,1,2,1);
  if(!host)return 1;
  poremc::HostGridConfig cfg;cfg.cell=cell;cfg.mesh={m,m,m};cfg.temperature=T;
  cfg.steep_interpolation=1;cfg.surrogate_cap=1e5;
  auto field=[&](){
    std::vector<double> raw(std::size_t(m)*m*m);
    for(int i=0;i<m;++i)for(int j=0;j<m;++j)for(int k=0;k<m;++k){
      const double phi=6.283185307179586 /m;
      const double wave=std::sin(phi*i)*std::cos(phi*j)+.01*std::sin(phi*k);
      raw[(i*m+j)*m+k]=750.*wave+(k==m-1 || k%16==0? 110000.:0);
    }
    return raw;
  };
  {
    poremc::HostGrid grid(cfg,host,3,zero,zero,q,field);
    constexpr int B=65536;
    std::mt19937_64 gen(2049);
    std::uniform_real_distribution<double> unif(-1.,2.);
    std::vector<double> xyz(9*B),out(B);
    for(int i=0;i<3*B;++i){
      double f[3]={unif(gen),unif(gen),unif(gen)};
      if(i%111==0) {f[0]=1.;f[1]=0.;f[2]=1.;}
      if(i%107==0)f[2]=std::nextafter(1.,0.);
      for(int d=0;d<3;++d)xyz[3*i+d]=f[0]*cell[d]+f[1]*cell[3+d]+f[2]*cell[6+d];
    }
    using Clock=std::chrono::steady_clock;
    grid.batch_energy(B,3,xyz.data(),0,out.data());
    const auto start=Clock::now();
    for(int i=0;i<3;++i)grid.batch_energy(B,3,xyz.data(),0,out.data());
    const double seconds=std::chrono::duration<double>(Clock::now()-start).count();
    std::cout<<std::setprecision(17)<<"{\"temperature_K\":"<<T<<",\"queries_s\":"<<9.*B/seconds<<",\"cubic\":"<<grid.stats().cubic_lookups<<",\"boltzmann\":"<<grid.stats().boltzmann_lookups<<",\"working_bytes\":"<<grid.stats().bytes<<",\"probes\":[";
    for(int i=0;i<1024;++i)std::cout<<(i?",":"")<<out[(i*61)%B];
    std::cout<<"]}\n";
  }
  poremc_environment_destroy(host);
}
