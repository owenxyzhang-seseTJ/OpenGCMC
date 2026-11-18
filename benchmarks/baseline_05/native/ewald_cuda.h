#ifndef POREMC_EWALD_CUDA_H
#define POREMC_EWALD_CUDA_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
// Native optional plug-in. k[3*modes], coefficient[modes] in kelvin.
void* poremc_ewald_cuda_create(size_t modes,const double* k,const double* coefficient,int device);
void poremc_ewald_cuda_destroy(void* handle);
const char* poremc_ewald_cuda_last_error(void);
// base is interleaved real/imaginary accepted S - exclusion + prefix.
int poremc_ewald_cuda_batch(void* handle,size_t batches,size_t atoms,
    const double* xyz,const double* charge,const double* base,double base_charge,
    double self_coefficient,double background_coefficient,double* output);
#ifdef __cplusplus
}
#endif
#endif
