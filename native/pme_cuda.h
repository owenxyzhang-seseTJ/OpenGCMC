#ifndef POREMC_PME_CUDA_H
#define POREMC_PME_CUDA_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct { size_t index; double value; } PoreMCPmeTerm;

void* poremc_pme_cuda_create(const int* mesh,const double* inverse,const double* influence,
                            const double* green,int device);
void poremc_pme_cuda_destroy(void* handle);
const char* poremc_pme_cuda_error(void);
const char* poremc_pme_cuda_device(void* handle);
int poremc_pme_cuda_refresh(void* handle,const double* mesh_charge,double* mesh_phi);
int poremc_pme_cuda_batch(void* handle,size_t batches,size_t atoms,const double* xyz,const double* charge,
                          size_t corrections,const PoreMCPmeTerm* terms,double constant,double* out);

#ifdef __cplusplus
}
#endif
#endif
