#include "pme_cuda.h"
#include "cuda_buffers.h"
#include <cufft.h>
#include <algorithm>
#include <memory>
#include <vector>

namespace {
using poremc_cuda::Buffer;
using poremc_cuda::check;
constexpr int threads=128;
thread_local std::string error_message;

void fft_check(cufftResult status,const char* operation) {
    if(status!=CUFFT_SUCCESS)
        throw std::runtime_error(std::string(operation)+": cuFFT error "+std::to_string(status));
}

size_t size_product(size_t a,size_t b) {
    if(b&&a>std::numeric_limits<size_t>::max()/b)throw std::overflow_error("CUDA PME dimensions overflow");
    return a*b;
}

struct Mesh {
    int dims[3],y_shift,x_shift;
    double inverse[9];
};

struct PmeGpu {
    int device=0;
    size_t points=0,spectrum_size=0,grid_limit=0;
    Mesh mesh{};
    cudaStream_t stream=nullptr;
    cufftHandle forward=0,reverse=0;
    std::string device_name;
    Buffer<double> phi,influence,green,xyz,charge,result;
    Buffer<cufftDoubleComplex> spectrum;
    Buffer<PoreMCPmeTerm> candidates,correction;
    bool ready=false;

    ~PmeGpu() {
        cudaSetDevice(device);
        if(forward)cufftDestroy(forward);
        if(reverse)cufftDestroy(reverse);
        if(stream)cudaStreamDestroy(stream);
    }

    unsigned blocks(size_t count) const {return static_cast<unsigned>(std::min(count,grid_limit));}
    void initialize(const int* dimensions,const double* inverse,const double* coefficients,
                    const double* real_green,int selected) {
        device=selected;
        check(cudaSetDevice(device),"select CUDA PME device");
        cudaDeviceProp properties{};
        check(cudaGetDeviceProperties(&properties,device),"query CUDA PME device");
        device_name=properties.name;
        grid_limit=properties.maxGridSize[0];
        std::copy_n(dimensions,3,mesh.dims);
        std::copy_n(inverse,9,mesh.inverse);
        const int nx=dimensions[0],ny=dimensions[1],nz=dimensions[2];
        for(int n=nz;n>1;n>>=1)++mesh.y_shift;
        mesh.x_shift=mesh.y_shift;
        for(int n=ny;n>1;n>>=1)++mesh.x_shift;
        const size_t planes=size_product(nx,ny);
        points=size_product(planes,nz);
        spectrum_size=size_product(planes,nz/2+1);
        check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"create CUDA PME stream");
        fft_check(cufftPlan3d(&forward,nx,ny,nz,CUFFT_D2Z),"plan forward PME FFT");
        fft_check(cufftPlan3d(&reverse,nx,ny,nz,CUFFT_Z2D),"plan inverse PME FFT");
        fft_check(cufftSetStream(forward,stream),"set forward PME FFT stream");
        fft_check(cufftSetStream(reverse,stream),"set inverse PME FFT stream");
        std::vector<double> compact(spectrum_size);
        for(size_t i=0;i<planes;++i)for(int k=0;k<=nz/2;++k)
            // cuFFT uses an unnormalized inverse. The remaining factor is
            // the derivative of the quadratic mesh energy with respect to charge.
            compact[i*(nz/2+1)+k]=2*coefficients[i*nz+k];
        put(influence,compact.data(),spectrum_size,"upload PME influence coefficients");
        put(green,real_green,points,"upload PME Green function");
        phi.reserve(points);spectrum.reserve(spectrum_size);
        check(cudaStreamSynchronize(stream),"finish PME initialization");
    }

    template<class T> void put(Buffer<T>& buffer,const T* host,size_t count,const char* operation) {
        buffer.reserve(count);
        if(count)check(cudaMemcpyAsync(buffer.data,host,count*sizeof(T),cudaMemcpyHostToDevice,stream),operation);
    }
};

__global__ void pme_influence(size_t size,cufftDoubleComplex* spectrum,const double* multiplier) {
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<size;i+=size_t(gridDim.x)*blockDim.x) {
        spectrum[i].x*=multiplier[i];
        spectrum[i].y*=multiplier[i];
    }
}

__global__ void pme_spread_trials(Mesh mesh,size_t queries,size_t atoms,const double* xyz,
                                  const double* charge,PoreMCPmeTerm* terms) {
    __shared__ int nodes[3][4];
    __shared__ double weights[3][4];
    for(size_t i=blockIdx.x;i<queries;i+=gridDim.x) {
        const int d=threadIdx.x;
        if(d<3) {
            const double* x=xyz+3*i;
            const double f=x[0]*mesh.inverse[d]+x[1]*mesh.inverse[3+d]+x[2]*mesh.inverse[6+d];
            const double u=(f-floor(f))*mesh.dims[d];
            const int base=int(u);
            const double t=u-base,one=1-t;
            weights[d][0]=one*one*one/6;
            weights[d][1]=(3*t*t*t-6*t*t+4)/6;
            weights[d][2]=(-3*t*t*t+3*t*t+3*t+1)/6;
            weights[d][3]=t*t*t/6;
            for(int b=0;b<4;++b)nodes[d][b]=(base+b-1+mesh.dims[d])&(mesh.dims[d]-1);
        }
        __syncthreads();
        const int a=d/16,b=(d/4)%4,c=d%4;
        const size_t node=(size_t(nodes[0][a])*mesh.dims[1]+nodes[1][b])*mesh.dims[2]+nodes[2][c];
        terms[64*i+d]={node,charge[i%atoms]*weights[0][a]*weights[1][b]*weights[2][c]};
        __syncthreads();
    }
}

__device__ size_t wrapped_difference(const Mesh& mesh,size_t a,size_t b) {
    const size_t dx=((a>>mesh.x_shift)-(b>>mesh.x_shift))&(mesh.dims[0]-1);
    const size_t dy=((a>>mesh.y_shift)-(b>>mesh.y_shift))&(mesh.dims[1]-1);
    return (dx<<mesh.x_shift)|(dy<<mesh.y_shift)|((a-b)&(mesh.dims[2]-1));
}

__global__ void pme_trial_energy(Mesh mesh,size_t batches,size_t nodes,const PoreMCPmeTerm* all,
                                 const double* green,const double* phi,size_t corrections,
                                 const PoreMCPmeTerm* extra,double constant,double* out) {
    __shared__ double sum[threads];
    const unsigned tid=threadIdx.x;
    for(size_t b=blockIdx.x;b<batches;b+=gridDim.x) {
        const auto* terms=all+b*nodes;
        double contribution=0;
        for(size_t i=tid;i<nodes;i+=blockDim.x) {
            const auto site=terms[i];
            if(site.value==0.)continue;
            double own=0.,cross=0.;
            // The real, symmetric Green function needs one triangle only.
            for(size_t j=0;j<i;++j)
                own+=green[wrapped_difference(mesh,site.index,terms[j].index)]*terms[j].value;
            for(size_t j=0;j<corrections;++j)
                cross+=green[wrapped_difference(mesh,site.index,extra[j].index)]*extra[j].value;
            contribution+=site.value*(phi[site.index]+site.value*green[0]+2*own+2*cross);
        }
        sum[tid]=contribution;
        __syncthreads();
        for(unsigned stride=threads/2;stride;stride/=2) {
            if(tid<stride)sum[tid]+=sum[tid+stride];
            __syncthreads();
        }
        if(!tid)out[b]=constant+sum[0];
        __syncthreads();
    }
}
}

extern "C" {
void* poremc_pme_cuda_create(const int* mesh,const double* inverse,const double* influence,
                            const double* green,int device) {
    try {
        auto state=std::make_unique<PmeGpu>();
        state->initialize(mesh,inverse,influence,green,device);
        return state.release();
    } catch(const std::exception& e) {error_message=e.what();return nullptr;}
}
void poremc_pme_cuda_destroy(void* handle) {delete static_cast<PmeGpu*>(handle);}
const char* poremc_pme_cuda_error(void) {return error_message.c_str();}
const char* poremc_pme_cuda_device(void* handle) {return static_cast<PmeGpu*>(handle)->device_name.c_str();}

int poremc_pme_cuda_refresh(void* handle,const double* mesh_charge,double* mesh_phi) {
    auto& g=*static_cast<PmeGpu*>(handle);
    try {
        g.ready=false;
        check(cudaSetDevice(g.device),"select CUDA PME device");
        g.put(g.phi,mesh_charge,g.points,"upload PME charge mesh");
        fft_check(cufftExecD2Z(g.forward,g.phi.data,g.spectrum.data),"forward PME FFT");
        pme_influence<<<g.blocks((g.spectrum_size-1)/threads+1),threads,0,g.stream>>>(
            g.spectrum_size,g.spectrum.data,g.influence.data);
        check(cudaGetLastError(),"apply PME influence function");
        fft_check(cufftExecZ2D(g.reverse,g.spectrum.data,g.phi.data),"inverse PME FFT");
        check(cudaMemcpyAsync(mesh_phi,g.phi.data,g.points*sizeof(double),cudaMemcpyDeviceToHost,g.stream),"read PME potential mesh");
        check(cudaStreamSynchronize(g.stream),"finish PME mesh refresh");
        g.ready=true;
        return 0;
    } catch(const std::exception& e) {cudaStreamSynchronize(g.stream);error_message=e.what();return -1;}
}

int poremc_pme_cuda_batch(void* handle,size_t batches,size_t atoms,const double* xyz,const double* charge,
                          size_t corrections,const PoreMCPmeTerm* terms,double constant,double* out) {
    auto& g=*static_cast<PmeGpu*>(handle);
    try {
        if(!batches)return 0;
        if(!g.ready||!atoms)throw std::logic_error("CUDA PME requires a refreshed mesh and molecular sites");
        check(cudaSetDevice(g.device),"select CUDA PME device");
        const size_t queries=size_product(batches,atoms);
        g.candidates.reserve(size_product(queries,64));
        g.result.reserve(batches);
        g.put(g.xyz,xyz,size_product(queries,3),"upload PME trial coordinates");
        g.put(g.charge,charge,atoms,"upload PME trial charges");
        g.put(g.correction,terms,corrections,"upload sparse PME charge corrections");
        pme_spread_trials<<<g.blocks(queries),64,0,g.stream>>>(g.mesh,queries,atoms,g.xyz.data,g.charge.data,g.candidates.data);
        check(cudaGetLastError(),"assign PME trial B-splines");
        pme_trial_energy<<<g.blocks(batches),threads,0,g.stream>>>(g.mesh,batches,64*atoms,g.candidates.data,
            g.green.data,g.phi.data,corrections,g.correction.data,constant,g.result.data);
        check(cudaGetLastError(),"evaluate PME trial energies");
        check(cudaMemcpyAsync(out,g.result.data,batches*sizeof(double),cudaMemcpyDeviceToHost,g.stream),"read PME trial energies");
        check(cudaStreamSynchronize(g.stream),"finish PME trial batch");
        return 0;
    } catch(const std::exception& e) {cudaStreamSynchronize(g.stream);error_message=e.what();return -1;}
}
}
