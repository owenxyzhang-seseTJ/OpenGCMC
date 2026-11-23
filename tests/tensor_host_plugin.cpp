// CPU test double for the optional plug-in ABI, used only in a temporary test directory.
// It lets CI exercise lattice upload, batch and selected-contribution plumbing on CPU.
#include "ewald_cuda.h"
#include "ewald_cuda_tensor.h"
#include <vector>
#include <complex>
#include <stdexcept>
#include <string>
using namespace poremc_cuda_tensor;
namespace {
thread_local std::string diagnostic;
struct Mock {
    std::vector<double> weights, kvector;
    Plan lattice;
    std::vector<Complex> resident, change;
    std::vector<Complex> evaluate(size_t n,const double* xyz,const double* q) const {
        std::vector<Complex> powers(n*lattice.axes.span), result(weights.size());
        if(lattice.tiles.empty()){
            for(size_t k=0;k<weights.size();++k){
                for(size_t a=0;a<n;++a){
                    const double angle=xyz[3*a]*kvector[3*k]+xyz[3*a+1]*kvector[3*k+1]+xyz[3*a+2]*kvector[3*k+2];
                    result[k].re+=std::cos(angle)*q[a];result[k].im+=std::sin(angle)*q[a];
                }
            }
            return result;
        }
        for(size_t a=0;a<n;++a)for(int d=0;d<3;++d)
            harmonics(lattice.axes,d,xyz+3*a,powers.data()+a*lattice.axes.span);
        for(const Tile& tile:lattice.tiles){
            for(size_t first=0;first<n;first+=32){
                Complex xy[32];
                const size_t count=std::min<size_t>(32,n-first);
                for(size_t j=0;j<count;++j){
                    const Complex* v=powers.data()+(first+j)*lattice.axes.span;
                    xy[j]=multiply(v[tile.ix],v[tile.iy]);
                    xy[j].re*=q[first+j];xy[j].im*=q[first+j];
                }
                for(int lane=0;lane<tile.count;++lane){
                    const int k=tile.first+lane;
                    for(size_t j=0;j<count;++j){
                        const Complex z=powers[(first+j)*lattice.axes.span+lattice.z_index[k]];
                        const Complex term=multiply(xy[j],z);
                        result[k].re+=term.re;result[k].im+=term.im;
                    }
                }
            }
        }
        return result;
    }
};
}
extern "C" {
void* poremc_ewald_cuda_create(size_t n,const double*k,const double*w,int){return new Mock{std::vector<double>(w,w+n),std::vector<double>(k,k+3*n),{}};}
void poremc_ewald_cuda_destroy(void*p){delete static_cast<Mock*>(p);}
const char* poremc_ewald_cuda_last_error(){return diagnostic.c_str();}
const char* poremc_ewald_cuda_kernel_name(void*){return "cuda-tensor";}
const char* poremc_ewald_cuda_device_name(void*){return "CPU ABI test double (no device)";}
int poremc_ewald_cuda_set_lattice(void*p,const double*h,const int*limits,const int*indices){
    Mock& g=*static_cast<Mock*>(p);
    try{g.lattice=plan(g.weights.size(),h,limits,indices);return 0;}
    catch(const std::exception& error){diagnostic=error.what();return -1;}
}
int poremc_ewald_cuda_structure(void*p,size_t n,const double*x,const double*q,double*out){
    const auto modes=static_cast<Mock*>(p)->evaluate(n,x,q);
    for(size_t k=0;k<modes.size();++k){out[2*k]=modes[k].re;out[2*k+1]=modes[k].im;}
    return 0;
}
int poremc_ewald_cuda_resident_import(void* p,const double* modes){
    auto& mock=*static_cast<Mock*>(p);
    mock.resident.resize(mock.weights.size());
    for(size_t i=0;i<mock.weights.size();++i)mock.resident[i]={modes[2*i],modes[2*i+1]};
    return 0;
}
int poremc_ewald_cuda_resident_export(void* p,double* modes){
    const auto& mock=*static_cast<Mock*>(p);
    for(size_t i=0;i<mock.weights.size();++i){
        modes[2*i]=mock.resident[i].re;modes[2*i+1]=mock.resident[i].im;
    }
    return 0;
}
int poremc_ewald_cuda_resident_trial(void* p,size_t n,const double* x,const double* q,double* out){
    auto& mock=*static_cast<Mock*>(p);
    mock.change=mock.evaluate(n,x,q);
    *out=0.;
    for(size_t i=0;i<mock.weights.size();++i)
        *out+=energy(mock.weights[i],mock.resident[i].re,mock.resident[i].im,mock.change[i]);
    return 0;
}
int poremc_ewald_cuda_resident_commit(void* p){
    auto& mock=*static_cast<Mock*>(p);
    for(size_t i=0;i<mock.weights.size();++i){
        mock.resident[i].re+=mock.change[i].re;mock.resident[i].im+=mock.change[i].im;
    }
    return 0;
}
int poremc_ewald_cuda_batch(void*p,size_t B,size_t n,const double*x,const double*q,const double*base,
                            double total,double self,double bg,double*out){
    const Mock& g=*static_cast<Mock*>(p);
    double sum=0.,sq=0.;for(size_t a=0;a<n;++a){sum+=q[a];sq+=q[a]*q[a];}
    for(size_t b=0;b<B;++b){
        auto s=g.evaluate(n,x+3*b*n,q);
        out[b]=self*sq+bg*(2*total*sum+sum*sum);
        for(size_t k=0;k<s.size();++k)out[b]+=energy(g.weights[k],base[2*k],base[2*k+1],s[k]);
    }
    return 0;
}
}
