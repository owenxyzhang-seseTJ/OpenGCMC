#ifndef POREMC_EWALD_STATE_H
#define POREMC_EWALD_STATE_H
#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace poremc {
// All electrostatic energies are U/k_B (kelvin); coordinates angstrom; q in e.
// Cell lattice vectors are rows. 0=direct Ewald, 1=cubic B-spline PME.
// Backend: 0=CPU, 1=auto (CUDA if available for direct), 2=required CUDA, 3=required FP64 lattice tensor CUDA.
struct ReciprocalConfig {
  std::array<double,9> cell{};
  double alpha=0.25;
  std::array<int,3> kmax{{6,6,6}}, mesh{{32,32,32}};
  int method=0, backend=0, threads=1;
  bool background=true;
  // Combine conjugate +/- k pairs without changing the finite Ewald sum.
  bool half_spectrum=true;
  // Optional physical wave-number cutoff (angstrom^-1), inside kmax box.
  // Zero retains every original rectangular mode. Applies to direct only.
  double k_cutoff=0.0;
  std::string cuda_library;
};
struct ChargeContribution {
  std::vector<std::complex<double>> modes;
  std::vector<std::size_t> indices;
  std::vector<double> values;
  double charge=0.0, squared_charge=0.0;
};
struct ReciprocalStats {
  std::uint64_t host_builds=0, contributions=0, candidate_batches=0, candidates=0;
  std::uint64_t accepted_commits=0, fft_transforms=0, gpu_batches=0;
  std::uint64_t modes=0, mesh_points=0;
  std::uint64_t gpu_selected_structures=0;
  std::uint64_t full_modes=0, host_grid_builds=0, host_field_evaluations=0;
};
class EwaldState {
 public:
  EwaldState(const ReciprocalConfig&, std::size_t host_atoms,
             const double* host_positions, const double* host_charge);
  ~EwaldState();
  EwaldState(const EwaldState&)=delete;
  EwaldState& operator=(const EwaldState&)=delete;
  ChargeContribution contribution(std::size_t atoms,const double* positions,const double* charge) const;
  // FP64 device tensor for a selected delayed trial, CPU unless explicitly requested.
  ChargeContribution selected_contribution(std::size_t atoms,const double* positions,const double* charge) const;
  // Direct Ewald rigid translation: S_new(k)=S_old(k) exp(i k.d).
  // No atom traversal; preserves charge and charge-square invariants.
  ChargeContribution translated_contribution(const ChargeContribution& old,
                                             const std::array<double,3>& displacement) const;
  // A missing/empty contribution means zero. Accumulate also supports sign=-1.
  void accumulate(ChargeContribution& target,const ChargeContribution& source,double sign=1.0) const;
  double insertion_energy(const ChargeContribution& added,
                          const ChargeContribution* excluded=nullptr,
                          const ChargeContribution* prefix=nullptr) const;
  // New minus old energy relative to the current state (it includes removed).
  double replacement_energy(const ChargeContribution& added, const ChargeContribution& removed) const;
  // Immutable host/guest reciprocal cross term only (no self/background).
  double host_insertion_energy(const ChargeContribution& added) const;
  double host_insertion_energy(std::size_t atoms,const double* positions,
                               const double* charge) const;
  // Unit-charge host potential at fractional nodes i/mesh; C-order layout.
  // Direct: exact finite Fourier values at every node via aliased FFT bins.
  // PME: exact interpolation of the immutable PME host potential.
  std::vector<double> host_potential_grid(const std::array<int,3>& mesh) const;
  // xyz layout [batches, atoms, 3]; every batch uses the same charge[atoms].
  // Evaluates relative to accepted state - excluded + prefix, without mutation.
  void batch_insertion(std::size_t batches,std::size_t atoms,const double* xyz,const double* charge,
                       const ChargeContribution* excluded,const ChargeContribution* prefix,double* output) const;
  // Commit exactly once after an accepted insertion/deletion/replacement.
  void commit(const ChargeContribution* added,const ChargeContribution* removed=nullptr);
  double total_host_relative_energy() const;
  std::string serialize() const;
  void restore(const std::string&);
  ReciprocalStats stats() const;
  const char* backend_name() const;
  const char* device_name() const;
  const ReciprocalConfig& config() const;
 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
};
}
#endif
