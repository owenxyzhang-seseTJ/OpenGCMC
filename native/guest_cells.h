#ifndef POREMC_GUEST_CELLS_H
#define POREMC_GUEST_CELLS_H

#include <cstddef>
#include <memory>

namespace poremc {
// Mutable neighbor index for the accepted guest configuration. A molecule's
// sites share an owner index. Cell order is reproducible from the checkpoint.
class GuestCells {
public:
  GuestCells(const double* cell, double cutoff, int coulomb, double alpha, int shifted);
  ~GuestCells();
  void reserve(std::size_t atoms, const double* xyz);
  void insert(std::size_t owner, std::size_t atoms, const double* xyz,
              const double* sigma, const double* epsilon, const double* charge);
  void erase(std::size_t owner, std::size_t atoms, const double* xyz);
  void batch(std::size_t batches, std::size_t atoms, const double* xyz,
             const double* sigma, const double* epsilon, const double* charge,
             std::size_t exclude_begin, std::size_t exclude_end, double* output) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}
#endif
