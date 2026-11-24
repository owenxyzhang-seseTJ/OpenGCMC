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
  // Exact half-spectrum algebra and optional reference-CBMC screening.
  int32_t ewald_half_spectrum, sampling_mode, host_grid_mesh[3];
  double ewald_k_cutoff, host_grid_cap, grid_root_bias;
  const char *host_grid_cache, *host_grid_key;
  int32_t host_grid_steep;
};
#ifdef __cplusplus
extern "C" {
#endif
size_t poremc_fast_config_size(void);
const char* poremc_fast_last_error(void);
void* poremc_fast_create(const struct PoreMCFastConfig* c);
// Component entries share the simulation settings in configs[0].
void* poremc_fast_create_components(const struct PoreMCFastConfig* configs,size_t count,
                                   const double* weights,int local,size_t pme_refresh,
                                   int background,int partial_regrowth,
                                   const char* lj_prefix,const char* electrostatic_path);
void poremc_fast_destroy(void* handle);
int poremc_fast_add(void* handle,const double* coordinates);
int poremc_fast_add_raw(void* handle,const double* coordinates);
int poremc_fast_add_component(void* handle,size_t species,const double* coordinates,int raw);
int poremc_fast_counts(void* handle,uint64_t* counts);
int poremc_fast_set_activities(void* handle,const double* activities);
int poremc_fast_export_particles(void* handle,uint64_t* species,double* xyz);
size_t poremc_fast_count(void* handle);
uint64_t poremc_fast_step_count(void* handle);
int poremc_fast_positions(void* handle,double* coordinates);
int poremc_fast_counters(void* handle,uint64_t* counters);
int poremc_fast_run(void* handle,uint64_t burnin,uint64_t steps,uint64_t sample_every,
                   uint64_t* counts,double* energies,int record_energy,double* timing);
// The count output is row-major [samples, components].
int poremc_fast_run_components(void* handle,uint64_t burnin,uint64_t steps,uint64_t sample_every,
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
int poremc_fast_surrogate_site_energies(void* handle,size_t exclude,size_t prefix_atoms,
                                      const double* prefix,size_t batches,const double* trials,double* energies);
int poremc_fast_surrogate_molecule_energies(void* handle,size_t exclude,size_t batches,
                                          const double* trials,double* energies);
int poremc_fast_short_range_enable(void* p,const char* library,size_t min_batch);
size_t poremc_fast_short_range_info(void* p,char* buffer,size_t capacity);
int poremc_fast_probe_sites(void* handle,size_t species,size_t exclude,size_t prefix_atoms,
                           const double* prefix,size_t batches,const double* trials,
                           int reference,double* energies);
int poremc_fast_probe_molecules(void* handle,size_t species,size_t exclude,size_t batches,
                               const double* trials,int reference,double* energies);
size_t poremc_fast_acceleration_info(void* handle,char* buffer,size_t capacity);
int poremc_fast_acceleration_counters(void* handle,uint64_t* output);
int poremc_fast_restore_acceleration_counters(void* handle,const uint64_t* input);
#ifdef __cplusplus
}
#endif
#endif
