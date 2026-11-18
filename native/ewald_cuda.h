#ifndef POREMC_EWALD_CUDA_H
#define POREMC_EWALD_CUDA_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
// Native optional plug-in. k[3*modes], coefficient[modes] in kelvin.
void* poremc_ewald_cuda_create(size_t modes,const double* k,const double* coefficient,int device);
// Optional tensor API. integer_modes lists the exact retained modes, in energy order.
int poremc_ewald_cuda_set_lattice(void* handle,const double* inverse_cell,const int* kmax,const int* integer_modes);
const char* poremc_ewald_cuda_kernel_name(void* handle);
const char* poremc_ewald_cuda_device_name(void* handle);
// Return one selected complete molecule in exact mode order, interleaved [K,2].
int poremc_ewald_cuda_structure(void* handle,size_t atoms,const double* xyz,const double* charge,double* structure);
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
