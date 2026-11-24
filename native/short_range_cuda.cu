// FP64 native real-space batches. The framework CSR and accepted guests reside
// on the device between calls. One block evaluates each trial atom.
#include "short_range_cuda.h"
#include "cuda_buffers.h"
#include "../poremc/cached_cuda.cu"
#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

namespace {
using poremc_cuda::Buffer;
using poremc_cuda::check;
constexpr unsigned block_size=128;
constexpr double coulomb_constant=167100.94689828736;
thread_local std::string pair_error;

struct Geometry {
    double cell[9], inverse[9];
    double cutoff, cutoff2, inverse_cutoff2, alpha, dsf_value, dsf_slope;
    int bins[3], diagonal, shifted, coulomb;
};

struct Parameters { double sigma, sqrt_epsilon, charge; };
struct PairEnergy { double framework, guests; };

size_t product(size_t left,size_t right) {
    if(left&&right>std::numeric_limits<size_t>::max()/left)
        throw std::overflow_error("CUDA short-range dimensions overflow");
    return left*right;
}

template<typename T>
void upload(Buffer<T>& buffer,const T* data,size_t count,cudaStream_t stream,const char* operation) {
    buffer.reserve(count);
    if(count)check(cudaMemcpyAsync(buffer.data,data,count*sizeof(T),cudaMemcpyHostToDevice,stream),operation);
}

struct GpuPairs {
    int device=0;
    cudaStream_t stream=nullptr;
    std::string device_name;
    size_t grid_limit=0, framework_sites=0, guest_sites=0;
    Geometry geometry{};
    Buffer<PoreMCCudaPairSite> framework, guests;
    Buffer<size_t> offsets;
    Buffer<uint32_t> neighbours;
    Buffer<double> xyz, result;
    Buffer<Parameters> parameters;
    Buffer<PairEnergy> atom_energy;
    std::vector<Parameters> host_parameters;
    bool indexed=false;

    ~GpuPairs() {
        cudaSetDevice(device);
        if(stream)cudaStreamDestroy(stream);
    }

    void initialize(const PoreMCCudaPairHost& host,int selected) {
        device=selected;
        check(cudaSetDevice(device),"select CUDA short-range device");
        cudaDeviceProp properties{};
        check(cudaGetDeviceProperties(&properties,device),"query CUDA short-range device");
        device_name=properties.name;
        grid_limit=static_cast<size_t>(properties.maxGridSize[0]);
        check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"create CUDA short-range stream");
        std::copy_n(host.cell,9,geometry.cell);
        std::copy_n(host.inverse,9,geometry.inverse);
        std::copy_n(host.bins,3,geometry.bins);
        geometry.cutoff=host.cutoff;
        geometry.cutoff2=host.cutoff*host.cutoff;
        geometry.inverse_cutoff2=1/geometry.cutoff2;
        geometry.alpha=host.alpha;
        geometry.dsf_value=host.dsf_value;
        geometry.dsf_slope=host.dsf_slope;
        geometry.diagonal=host.diagonal;
        geometry.shifted=host.shifted;
        geometry.coulomb=host.coulomb;
        upload(framework,host.coordinates,host.sites,stream,"upload CUDA framework");
        if(host.query_cells) {
            upload(offsets,host.offsets,host.query_cells+1,stream,"upload framework neighbour offsets");
            upload(neighbours,host.neighbours,host.neighbour_entries,stream,"upload framework neighbours");
            indexed=true;
        }
        check(cudaStreamSynchronize(stream),"finish framework upload");
        framework_sites=host.sites;
    }
};

__device__ inline double site_energy(const Geometry& g,const double* f,const Parameters& probe,
                                     const PoreMCCudaPairSite& site) {
    const double r2=poremc_cuda::periodic_r2(f[0]-site.fraction[0],f[1]-site.fraction[1],
                                             f[2]-site.fraction[2],g.cell,g.diagonal);
    if(!(r2<g.cutoff2))return 0.;
    const double s=.5*(probe.sigma+site.sigma),s2=s*s;
    const double e4=4*probe.sqrt_epsilon*site.sqrt_epsilon;
    const double c=s2*g.inverse_cutoff2,c3=c*c*c;
    const double shift=g.shifted?e4*c3*(c3-1):0.;
    const double force=g.shifted==2?6*e4*c3*(1-2*c3):0.;
    return poremc_cuda::pair_value(r2,s2,e4,coulomb_constant*probe.charge*site.charge,
                                    shift,force,g.coulomb,g.cutoff,g.alpha,g.dsf_value,g.dsf_slope);
}

__global__ void pair_atoms(Geometry geometry,size_t queries,size_t atoms,const double* xyz,
                          const Parameters* parameters,size_t frameworks,const PoreMCCudaPairSite* framework,
                          const size_t* offsets,const uint32_t* neighbours,
                          size_t guests,const PoreMCCudaPairSite* mobile,size_t exclude_begin,size_t exclude_end,
                          PairEnergy* partial) {
    __shared__ double host_sum[block_size],guest_sum[block_size];
    const unsigned tid=threadIdx.x;
    for(size_t i=blockIdx.x;i<queries;i+=gridDim.x) {
        const double* x=xyz+3*i;
        const double f[]={poremc_cuda::fractional_component(x,geometry.inverse,0),
                          poremc_cuda::fractional_component(x,geometry.inverse,1),
                          poremc_cuda::fractional_component(x,geometry.inverse,2)};
        const Parameters params=parameters[i%atoms];
        size_t begin=0,end=frameworks;
        if(offsets) {
            int cell[3];
            for(int d=0;d<3;++d)cell[d]=min(int(f[d]*geometry.bins[d]),geometry.bins[d]-1);
            const size_t index=(size_t(cell[0])*geometry.bins[1]+cell[1])*geometry.bins[2]+cell[2];
            begin=offsets[index];
            end=offsets[index+1];
        }
        double h=0.,m=0.;
        for(size_t j=begin+tid;j<end;j+=blockDim.x)
            h+=site_energy(geometry,f,params,framework[offsets?neighbours[j]:j]);
        for(size_t j=tid;j<guests;j+=blockDim.x)
            if(j<exclude_begin||j>=exclude_end)m+=site_energy(geometry,f,params,mobile[j]);
        host_sum[tid]=h;
        guest_sum[tid]=m;
        __syncthreads();
        for(unsigned stride=block_size/2;stride;stride/=2) {
            if(tid<stride) {
                host_sum[tid]+=host_sum[tid+stride];
                guest_sum[tid]+=guest_sum[tid+stride];
            }
            __syncthreads();
        }
        if(!tid)partial[i]={host_sum[0],guest_sum[0]};
        __syncthreads();
    }
}

__global__ void sum_pairs(size_t batches,size_t atoms,const PairEnergy* partial,double* energies) {
    for(size_t b=size_t(blockIdx.x)*blockDim.x+threadIdx.x;b<batches;b+=size_t(gridDim.x)*blockDim.x) {
        double h=0.,m=0.;
        for(size_t a=0;a<atoms;++a) {
            const auto value=partial[b*atoms+a];
            h+=value.framework;
            m+=value.guests;
        }
        energies[b]=h+m;
    }
}
}

extern "C" {
void* poremc_pair_cuda_create(const PoreMCCudaPairHost* host,int device) {
    try {
        if(!host)throw std::invalid_argument("null CUDA framework");
        auto pairs=std::make_unique<GpuPairs>();
        pairs->initialize(*host,device);
        return pairs.release();
    } catch(const std::exception& error) {
        pair_error=error.what();
        return nullptr;
    }
}

void poremc_pair_cuda_destroy(void* context) { delete static_cast<GpuPairs*>(context); }
const char* poremc_pair_cuda_error() { return pair_error.c_str(); }
const char* poremc_pair_cuda_device(void* context) {
    return static_cast<GpuPairs*>(context)->device_name.c_str();
}

int poremc_pair_cuda_upload_guests(void* context,size_t sites,const PoreMCCudaPairSite* coords) {
    auto& g=*static_cast<GpuPairs*>(context);
    try {
        check(cudaSetDevice(g.device),"select CUDA guest device");
        if(sites>g.guests.capacity) {
            const auto capacity=g.guests.capacity;
            g.guests.reserve(capacity<=std::numeric_limits<size_t>::max()/2?std::max(sites,2*capacity):sites);
        }
        upload(g.guests,coords,sites,g.stream,"upload accepted guests");
        check(cudaStreamSynchronize(g.stream),"finish accepted guest upload");
        g.guest_sites=sites;
        return 0;
    } catch(const std::exception& error) {
        cudaStreamSynchronize(g.stream);
        pair_error=error.what();
        return -1;
    }
}

int poremc_pair_cuda_energy(void* context,size_t batches,size_t atoms,const double* xyz,
                           const double* sigma,const double* epsilon,const double* charge,
                           size_t exclude_begin,size_t exclude_end,int parts,double* out) {
    auto& g=*static_cast<GpuPairs*>(context);
    try {
        if(!batches)return 0;
        const size_t hosts=parts&POREMC_PAIR_FRAMEWORK?g.framework_sites:0;
        const size_t guests=parts&POREMC_PAIR_GUESTS?g.guest_sites:0;
        if(exclude_begin>exclude_end||exclude_end>guests)throw std::invalid_argument("invalid CUDA guest exclusion span");
        if(!atoms||(!hosts&&!guests)) {std::fill_n(out,batches,0.);return 0;}
        const size_t queries=product(batches,atoms);
        check(cudaSetDevice(g.device),"select CUDA short-range device");
        g.host_parameters.resize(atoms);
        for(size_t a=0;a<atoms;++a)g.host_parameters[a]={sigma[a],std::sqrt(epsilon[a]),charge[a]};
        upload(g.xyz,xyz,product(3,queries),g.stream,"upload short-range trial coordinates");
        upload(g.parameters,g.host_parameters.data(),atoms,g.stream,"upload short-range trial parameters");
        g.atom_energy.reserve(queries);
        g.result.reserve(batches);
        const size_t* offsets=hosts&&g.indexed?g.offsets.data:nullptr;
        pair_atoms<<<static_cast<unsigned>(std::min(queries,g.grid_limit)),block_size,0,g.stream>>>(
            g.geometry,queries,atoms,g.xyz.data,g.parameters.data,hosts,g.framework.data,offsets,g.neighbours.data,
            guests,g.guests.data,exclude_begin,exclude_end,g.atom_energy.data);
        check(cudaGetLastError(),"launch CUDA short-range pairs");
        const auto blocks=static_cast<unsigned>(std::min((batches-1)/block_size+1,g.grid_limit));
        sum_pairs<<<blocks,block_size,0,g.stream>>>(batches,atoms,g.atom_energy.data,g.result.data);
        check(cudaGetLastError(),"launch CUDA short-range reduction");
        check(cudaMemcpyAsync(out,g.result.data,batches*sizeof(double),cudaMemcpyDeviceToHost,g.stream),
              "download short-range energies");
        check(cudaStreamSynchronize(g.stream),"finish CUDA short-range batch");
        return 0;
    } catch(const std::exception& error) {
        cudaStreamSynchronize(g.stream);
        pair_error=error.what();
        return -1;
    }
}
}
