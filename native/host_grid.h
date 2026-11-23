#ifndef POREMC_HOST_GRID_H
#define POREMC_HOST_GRID_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace poremc {

// A deterministic reference potential for delayed-acceptance CBMC. The
// exact target Hamiltonian is evaluated separately by the sampler.
// Row lattice vectors, Cartesian positions in Angstrom, energies U/k_B in K.
struct HostGridConfig {
  std::array<double,9> cell{};
  std::array<int,3> mesh{{32,32,32}};
  double steep_limit=1e6;
  // Positive: finite capped reference, with trilinear interpolation in
  // steep cells. Zero: uncapped grid, falling back to exact host evaluation.
  // This is NEVER an upper bound on the target physical energy.
  double surrogate_cap=1e5;
  // Steep cells: 0=linear energy; 1=linear Boltzmann weight. The latter
  // prevents a high capped corner from spreading a repulsive barrier across
  // a coarse cell. Temperature affects interpolation, never cached fields.
  int steep_interpolation=0;
  double temperature=300.;
  int threads=1;
  std::string cache_path, cache_key;
  std::string lj_prefix, electrostatic_path;
};

struct HostGridStats {
  std::uint64_t builds=0, cache_loads=0, cache_writes=0;
  std::uint64_t lookups=0, cubic_lookups=0, linear_lookups=0, boltzmann_lookups=0, exact_fallbacks=0;
  std::uint64_t mesh_points=0, lj_fields=0, site_types=0, bytes=0, steep_cells=0;
  std::uint64_t raw_field_builds=0, raw_field_loads=0;
};

using HostReciprocalGridBuilder=std::function<std::vector<double>()>;
using HostReciprocalEnergy=std::function<double(std::size_t,const double*,const double*)>;

class HostGrid {
 public:
  // host_environment is borrowed. Builder returns the host-only reciprocal
  // potential for a UNIT charge at grid nodes, C-order (ix*My+iy)*Mz+iz.
  // reciprocal_exact also returns host-guest cross energy only, with no
  // guest self interaction or net-charge background.
  HostGrid(const HostGridConfig&,void* host_environment,std::size_t atoms,
           const double* sigma,const double* epsilon,const double* charge,
           HostReciprocalGridBuilder reciprocal_grid_builder={},
           HostReciprocalEnergy reciprocal_exact={});
  ~HostGrid();
  HostGrid(const HostGrid&)=delete;
  HostGrid& operator=(const HostGrid&)=delete;

  // Every trial contains atoms contiguous molecular sites from first_site.
  // Typical calls: a full molecule with first_site=0, or one CBMC site i.
  void batch_energy(std::size_t batches,std::size_t atoms,const double* xyz,
                    std::size_t first_site,double* output) const;
  double site_energy(std::size_t site,const double* xyz) const;
  HostGridStats stats() const;
  const HostGridConfig& config() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
};
}
#endif
