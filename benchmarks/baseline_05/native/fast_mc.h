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
  // Long-range configuration. Coulomb 2=Ewald, 3=PME; both use unshifted
  // Ewald real space in the short-range kernel. Method 0=direct, 1=PME.
  int32_t ewald_kmax[3], pme_mesh[3], reciprocal_method, reciprocal_backend;
  uint64_t host_atoms;
  const double *host_positions, *host_charge;
  const char *cuda_library;
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
size_t poremc_fast_longrange_state(void* handle,char* buffer,size_t capacity);
int poremc_fast_restore_longrange(void* handle,const char* state);
size_t poremc_fast_longrange_info(void* handle,char* buffer,size_t capacity);
// Read-only physical-state probes; exclude=SIZE_MAX means no excluded molecule.
int poremc_fast_site_energies(void* handle,size_t exclude,size_t prefix_atoms,
                            const double* prefix,size_t batches,const double* trials,double* energies);
int poremc_fast_molecule_energies(void* handle,size_t exclude,size_t batches,
                                const double* trials,double* energies);
#ifdef __cplusplus
}
#endif
#endif
