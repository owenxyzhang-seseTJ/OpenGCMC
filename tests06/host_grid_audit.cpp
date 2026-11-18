// Independent small host-grid checks. This is not a GCMC timing benchmark.
#include "host_grid.h"
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>
extern "C" {
void* poremc_environment_create(std::size_t,const double*,const double*,const double*,const double*,const double*,double,int,double,int,int,int);
void poremc_environment_destroy(void*);
}
constexpr double pi=3.14159265358979323846,CK=167100.94689828736;
const std::array<double,9> H{{24,0,0,3,26,0,1,2,28}};
std::array<double,3> cart(double x,double y,double z){return {{24*x+3*y+z,26*y+2*z,28*z}};}
double potential(const double* r){
  const double z=r[2]/28,y=(r[1]-2*z)/26,x=(r[0]-3*y-z)/24;
  return 10*std::cos(2*pi*x)+4*std::sin(4*pi*y)+3*std::cos(2*pi*(x-z));
}
std::vector<double> field(int n){
  std::vector<double> f(std::size_t(n)*n*n);
  for(int x=0;x<n;++x)for(int y=0;y<n;++y)for(int z=0;z<n;++z){auto r=cart(double(x)/n,double(y)/n,double(z)/n);f[(std::size_t(x)*n+y)*n+z]=potential(r.data());}
  return f;
}
double exact_rec(std::size_t n,const double* r,const double* q){double v=0;for(std::size_t a=0;a<n;++a)v+=q[a]*potential(r+3*a);return v;}
void check(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
int main(int argc,char**argv){
  try{
    check(argc==2,"cache path required");
    const double zeros[3]={0,0,0};
    void* empty=poremc_environment_create(0,zeros,zeros,zeros,zeros,H.data(),9,2,.28,1,2,1);
    check(empty,"empty host creation failed");
    double s[2]={0,0},e[2]={0,0},q[2]={1,-.5};
    std::array<double,3> errors{};double node_error=0,periodic_error=0,batch_error=0;
    for(int resolution=0;resolution<3;++resolution){
      const int n=16<<resolution;poremc::HostGridConfig cfg;cfg.cell=H;cfg.mesh={{n,n,n}};
      poremc::HostGrid grid(cfg,empty,2,s,e,q,[n]{return field(n);},exact_rec);
      for(int i=0;i<300;++i){
        auto r=cart(std::fmod(.371*i+.123,1.),std::fmod(.227*i+.477,1.),std::fmod(.173*i+.331,1.));
        const double got=grid.site_energy(0,r.data());errors[resolution]=std::max(errors[resolution],std::abs(got-potential(r.data())));
        auto shifted=r;for(int d=0;d<3;++d)shifted[d]+=3*H[d]-2*H[3+d]+H[6+d];
        periodic_error=std::max(periodic_error,std::abs(got-grid.site_energy(0,shifted.data())));
        std::array<double,6> both{{r[0],r[1],r[2],r[0],r[1],r[2]}};double batch;
        grid.batch_energy(1,2,both.data(),0,&batch);batch_error=std::max(batch_error,std::abs(batch-.5*got));
        auto node=cart(double(i%n)/n,double((i/3)%n)/n,double((i/7)%n)/n);
        node_error=std::max(node_error,std::abs(grid.site_energy(0,node.data())-potential(node.data())));
      }
      check(grid.stats().lj_fields==1&&grid.stats().site_types==2,"field sharing failed");
    }
    check(errors[1]<errors[0]&&errors[2]<errors[1],"grid refinement did not converge");
    check(node_error<1e-11&&periodic_error<1e-11&&batch_error<1e-11,"grid nodes, periodicity, or site indexing failed");

    poremc::HostGridConfig cfg;cfg.cell=H;cfg.mesh={{32,32,32}};cfg.cache_path=argv[1];cfg.cache_key="independent-analytic-fixture-v1";
    std::filesystem::remove(cfg.cache_path);int builds=0;double first=0,second=0;
    auto r=cart(.183,.278,.877);
    {poremc::HostGrid grid(cfg,empty,2,s,e,q,[&]{++builds;return field(32);},exact_rec);first=grid.site_energy(1,r.data());check(grid.stats().cache_writes==1,"cache not written");}
    {poremc::HostGrid grid(cfg,empty,2,s,e,q,[]{throw std::runtime_error("cache hit rebuilt Fourier field");return std::vector<double>{};},exact_rec);second=grid.site_energy(1,r.data());check(grid.stats().cache_loads==1,"cache not loaded");}
    check(first==second&&builds==1,"cache changed field or failed lazy build");
    {std::fstream f(cfg.cache_path,std::ios::binary|std::ios::in|std::ios::out);f.seekg(-1,std::ios::end);char c;f.read(&c,1);c^=1;f.seekp(-1,std::ios::end);f.write(&c,1);}
    {poremc::HostGrid grid(cfg,empty,2,s,e,q,[&]{++builds;return field(32);},exact_rec);check(grid.stats().builds==1&&grid.site_energy(1,r.data())==first,"bad cache not rebuilt");}
    check(builds==2,"cache integrity check failed");std::filesystem::remove(cfg.cache_path);

    // Independent direct weighted partition sum in a deliberately steep
    // cell. The two temperatures must reuse the same raw field cache.
    const std::array<double,8> corner{{100000,600,900,-500,350,2200,100000,-2500}};
    auto steep_field=[&]{std::vector<double> f(32*32*32,100000.);
      for(int a=0;a<2;++a)for(int b=0;b<2;++b)for(int c=0;c<2;++c)f[(a*32+b)*32+c]=corner[(a*2+b)*2+c];return f;};
    cfg.cache_key="independent-boltzmann-fixture-v1";cfg.steep_interpolation=1;
    double boltzmann_error=0,boltzmann_node_error=0;std::array<double,2> thermal_values{};
    int thermal_builds=0;std::uint64_t boltzmann_lookups=0;
    for(int k=0;k<2;++k){
      cfg.temperature=k?900.:300.;
      poremc::HostGrid grid(cfg,empty,1,s,e,q,[&]{++thermal_builds;return steep_field();},exact_rec);
      auto position=cart(.2/32,.3/32,.4/32);const double got=grid.site_energy(0,position.data());
      double sum=0.;for(int a=0;a<2;++a)for(int b=0;b<2;++b)for(int c=0;c<2;++c){
        double weight=(a?.2:.8)*(b?.3:.7)*(c?.4:.6);sum+=weight*std::exp(-corner[(a*2+b)*2+c]/cfg.temperature);
      }
      const double expected=-cfg.temperature*std::log(sum);
      boltzmann_error=std::max(boltzmann_error,std::abs(got-expected));thermal_values[k]=got;
      boltzmann_lookups+=grid.stats().boltzmann_lookups;
      if(k)check(grid.stats().cache_loads==1,"temperature change rebuilt raw fields");
    }
    // At a grid node only one weight is nonzero. A minimum over all eight
    // energies would make the partition underflow at this low temperature.
    cfg.temperature=.1;
    {poremc::HostGrid grid(cfg,empty,1,s,e,q,[]{throw std::runtime_error("temperature-only change rebuilt grid");return std::vector<double>{};},exact_rec);
      boltzmann_node_error=std::abs(grid.site_energy(0,zeros)-corner[0]);
      boltzmann_lookups+=grid.stats().boltzmann_lookups;
      for(int i=0;i<100;++i){auto p=cart((i+.5)/3200.,.3/32,.4/32);check(std::isfinite(grid.site_energy(0,p.data())),"Boltzmann surrogate lost finite support");}
    }
    check(thermal_builds==1&&thermal_values[0]!=thermal_values[1]&&boltzmann_error<1e-9&&boltzmann_node_error==0.,"Boltzmann formula, node limit or thermal cache reuse failed");
    std::filesystem::remove(cfg.cache_path);cfg.steep_interpolation=0;cfg.temperature=300.;
    poremc_environment_destroy(empty);

    double hs=3.1,he=60,hq=.05;
    void* host=poremc_environment_create(1,zeros,&hs,&he,&hq,H.data(),9,2,.28,1,2,1);
    check(host,"host creation failed");double gs=3.2,ge=40,gq=.2,max_cap=0,max_fallback_relative=0;
    std::uint64_t linears=0,fallbacks=0;
    cfg.cache_path.clear();cfg.surrogate_cap=1e5;
    {poremc::HostGrid grid(cfg,host,1,&gs,&ge,&gq,[]{return field(32);},exact_rec);
      for(int i=0;i<100;++i){double pos[3]={.01*i,.003*i,.002*i};double u=grid.site_energy(0,pos);check(std::isfinite(u),"capped surrogate not finite at overlap");max_cap=std::max(max_cap,std::abs(u));}
      linears=grid.stats().linear_lookups;check(linears>0&&max_cap<=cfg.surrogate_cap*(1+1e-12),"positive-weight capped interpolation failed");
    }
    cfg.surrogate_cap=0.;
    {poremc::HostGrid grid(cfg,host,1,&gs,&ge,&gq,[]{return field(32);},exact_rec);
      for(int i=0;i<20;++i){double pos[3]={.3+.02*i,.1,.05};const double distance=std::sqrt(pos[0]*pos[0]+pos[1]*pos[1]+pos[2]*pos[2]);
        const double sigma=.5*(hs+gs),epsilon=std::sqrt(he*ge),sr6=std::pow(sigma/distance,6),cut6=std::pow(sigma/9,6);
        const double expected=4*epsilon*(sr6*(sr6-1)-cut6*(cut6-1))+CK*hq*gq*std::erfc(.28*distance)/distance+gq*potential(pos);
        max_fallback_relative=std::max(max_fallback_relative,std::abs(grid.site_energy(0,pos)-expected)/std::abs(expected));
      }
      fallbacks=grid.stats().exact_fallbacks;check(fallbacks==20&&max_fallback_relative<3e-12,"exact steep-stencil fallback disagrees with analytic pair reference");
    }
    poremc_environment_destroy(host);
    std::cout<<std::setprecision(17)<<"{\"passed\":true,\"analytic_grid_max_errors_K\":["<<errors[0]<<","<<errors[1]<<","<<errors[2]
      <<"],\"node_error_K\":"<<node_error<<",\"periodic_error_K\":"<<periodic_error<<",\"batch_error_K\":"<<batch_error
      <<",\"cache_reuse_and_integrity_passed\":true,\"linear_overlap_lookups\":"<<linears<<",\"maximum_capped_reference_K\":"<<max_cap
      <<",\"exact_fallbacks\":"<<fallbacks<<",\"fallback_max_relative_error\":"<<max_fallback_relative
      <<",\"boltzmann_formula_error_K\":"<<boltzmann_error<<",\"boltzmann_node_error_K\":"<<boltzmann_node_error
      <<",\"boltzmann_lookups\":"<<boltzmann_lookups<<",\"boltzmann_temperature_values_K\":["<<thermal_values[0]<<","<<thermal_values[1]
      <<"],\"temperature_independent_cache_reuse_passed\":true}\n";
  }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}
}
