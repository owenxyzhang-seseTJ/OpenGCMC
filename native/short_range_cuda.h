#ifndef POREMC_SHORT_RANGE_CUDA_H
#define POREMC_SHORT_RANGE_CUDA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Site positions are fractional and wrapped into the primary unit cell.
typedef struct {
    double fraction[3];
    double sigma, sqrt_epsilon, charge;
} PoreMCCudaPairSite;

// The plug-in copies the immutable CPU neighbour plan at construction.
typedef struct {
    double cell[9], inverse[9];
    double cutoff, alpha, dsf_value, dsf_slope;
    int bins[3], diagonal, shifted, coulomb;
    size_t sites, query_cells, neighbour_entries;
    const PoreMCCudaPairSite* coordinates;
    const size_t* offsets;
    const uint32_t* neighbours;
} PoreMCCudaPairHost;

enum { POREMC_PAIR_FRAMEWORK = 1, POREMC_PAIR_GUESTS = 2 };

typedef struct {
    const double* energy;
    const double* weight;
    const uint8_t* steep;
} PoreMCCudaGridField;

// Borrowed packed reference fields, copied to the device when registered.
typedef struct {
    int mesh[3], boltzmann;
    double inverse[9], temperature;
    size_t sites, types;
    const size_t* site_type;
    const PoreMCCudaGridField* fields;
} PoreMCCudaHostGrid;

void* poremc_pair_cuda_create(const PoreMCCudaPairHost* host, int device);
void poremc_pair_cuda_destroy(void* context);
const char* poremc_pair_cuda_error(void);
const char* poremc_pair_cuda_device(void* context);
int poremc_pair_cuda_upload_guests(void* context, size_t sites, const PoreMCCudaPairSite* guests);
int poremc_pair_cuda_update_guests(void* context, size_t sites, const PoreMCCudaPairSite* guests,
                                  size_t begin, size_t end, size_t* uploaded);
int poremc_pair_cuda_add_grid(void* context, const PoreMCCudaHostGrid* grid, size_t* index);
int poremc_pair_cuda_grid_energy(void* context, size_t grid, size_t first,
                                size_t batches, size_t atoms, const double* xyz,
                                const double* sigma, const double* epsilon, const double* charge,
                                size_t exclude_begin, size_t exclude_end, int parts, double* out);
int poremc_pair_cuda_energy(void* context, size_t batches, size_t atoms, const double* xyz,
                           const double* sigma, const double* epsilon, const double* charge,
                           size_t exclude_begin, size_t exclude_end, int parts, double* out);

// Dynamic-loader bridge provided by the CPU library. The existing Environment
// owns its host data, and is only borrowed during create.
void* poremc_short_cuda_create(void* environment, const char* library);
void poremc_short_cuda_destroy(void* context);
const char* poremc_short_cuda_device(void* context);
int poremc_short_cuda_upload_guests(void* context, size_t sites, const PoreMCCudaPairSite* guests);
int poremc_short_cuda_update_guests(void* context, size_t sites, const PoreMCCudaPairSite* guests,
                                   size_t begin, size_t end, size_t* uploaded);
int poremc_short_cuda_has_grid(void* context);
int poremc_short_cuda_add_grid(void* context, const PoreMCCudaHostGrid* grid, size_t* index);
int poremc_short_cuda_grid_energy(void* context, size_t grid, size_t first,
                                 size_t batches, size_t atoms, const double* xyz,
                                 const double* sigma, const double* epsilon, const double* charge,
                                 size_t exclude_begin, size_t exclude_end, int parts, double* out);
int poremc_short_cuda_energy(void* context, size_t batches, size_t atoms, const double* xyz,
                            const double* sigma, const double* epsilon, const double* charge,
                            size_t exclude_begin, size_t exclude_end, int parts, double* out);

#ifdef __cplusplus
}
#endif
#endif
