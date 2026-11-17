#ifndef POREMC_FAST_MC_H
#define POREMC_FAST_MC_H
#include <stddef.h>
#include <stdint.h>
// C ABI configuration; all arrays are COPIED by poremc_fast_create.
// Host handle is borrowed and must outlive FastMC. Cell lattice vectors are rows.
struct PoreMCFastConfig {
  uint64_t atoms, torsions, max_particles, seed;
  double temperature, activity, cutoff, alpha, max_translation, max_rotation;
  double scale14_lj, scale14_coulomb;
  int32_t shifted, coulomb, algorithm, threads, rigid;
  double cell[9], move_weights[5];
  const double *sigma, *epsilon, *charge, *bond_lengths, *angles, *rigid_positions;
  const double *bond_ranges, *angle_ranges, *bond_constants, *angle_constants;
  const int32_t *parents, *angle_refs, *dihedral_refs, *trials, *exclusions;
  const int32_t *torsion_indices, *torsion_periodicity;
  const double *torsion_k, *torsion_phase;
  void *host;
};
#ifdef __cplusplus
extern "C" {
#endif
size_t poremc_fast_config_size(void);
const char* poremc_fast_last_error(void);
void* poremc_fast_create(const struct PoreMCFastConfig* c);
void poremc_fast_destroy(void* handle);
int poremc_fast_add(void* handle,const double* coordinates);
int poremc_fast_add_raw(void* handle,const double* coordinates);
size_t poremc_fast_count(void* handle);
uint64_t poremc_fast_step_count(void* handle);
int poremc_fast_positions(void* handle,double* coordinates);
int poremc_fast_counters(void* handle,uint64_t* counters);
int poremc_fast_run(void* handle,uint64_t burnin,uint64_t steps,uint64_t sample_every,
                   uint64_t* counts,double* energies,int record_energy,double* timing);
int poremc_fast_total_energy(void* handle,double* energy);
size_t poremc_fast_rng(void* handle,char* buffer,size_t capacity);
int poremc_fast_restore_bookkeeping(void* handle,const char* rng,uint64_t steps,const uint64_t* counters);
#ifdef __cplusplus
}
#endif
#endif
