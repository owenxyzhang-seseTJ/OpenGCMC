#ifndef POREMC_CUDA_BUFFERS_H
#define POREMC_CUDA_BUFFERS_H

#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>
#include <string>

namespace poremc_cuda {
inline void check(cudaError_t status,const char* operation){
    if(status!=cudaSuccess)throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(status));
}

template<typename T> struct Buffer {
    T* data=nullptr;
    size_t capacity=0;

    Buffer()=default;
    Buffer(const Buffer&)=delete;
    Buffer& operator=(const Buffer&)=delete;
    ~Buffer(){if(data)cudaFree(data);}

    void reserve(size_t count){
        if(count<=capacity)return;
        if(count>std::numeric_limits<size_t>::max()/sizeof(T))throw std::overflow_error("CUDA buffer too large");
        T* next=nullptr;
        check(cudaMalloc(reinterpret_cast<void**>(&next),count*sizeof(T)),"cudaMalloc");
        if(data)cudaFree(data);
        data=next;
        capacity=count;
    }
};
}
#endif
