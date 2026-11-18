// Lattice plan and FP64 arithmetic shared by the CUDA contraction and its host oracle.
#ifndef POREMC_EWALD_CUDA_TENSOR_H
#define POREMC_EWALD_CUDA_TENSOR_H
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#ifdef __CUDACC__
#define POREMC_TENSOR_HD __host__ __device__
#else
#define POREMC_TENSOR_HD
#endif

namespace poremc_cuda_tensor {
constexpr int warp_width = 32;
struct Complex { double re, im; };
POREMC_TENSOR_HD inline Complex multiply(Complex a, Complex b) {
    return {a.re*b.re-a.im*b.im, a.re*b.im+a.im*b.re};
}
struct Axes {
    double inverse[9];
    int limit[3], offset[3], span;
};
struct Tile {
    int first, count, ix, iy;
};
struct Plan {
    Axes axes{};
    std::vector<Tile> tiles;
    std::vector<int> z_index;
};
inline Plan plan(std::size_t count, const double* inverse, const int* limits, const int* modes) {
    if (!inverse || !limits || !modes || count==0 || count>std::numeric_limits<int>::max())
        throw std::invalid_argument("invalid CUDA reciprocal lattice dimensions");
    Plan p;
    std::copy(inverse,inverse+9,p.axes.inverse);
    for (int d=0;d<3;++d) {
        if (limits[d]<0 || limits[d]>(std::numeric_limits<int>::max()-p.axes.span-1)/2)
            throw std::invalid_argument("invalid CUDA harmonic table size");
        p.axes.limit[d]=limits[d];
        p.axes.offset[d]=p.axes.span+limits[d];  // address of harmonic zero for this axis
        p.axes.span+=2*limits[d]+1;
    }
    p.z_index.reserve(count);
    for (std::size_t n=0;n<count;++n) {
        for (int d=0;d<3;++d)
            if (modes[3*n+d]<-limits[d] || modes[3*n+d]>limits[d])
                throw std::invalid_argument("reciprocal mode outside CUDA harmonic table");
        const int x=p.axes.offset[0]+modes[3*n], y=p.axes.offset[1]+modes[3*n+1];
        if (p.tiles.empty() || p.tiles.back().count==warp_width || p.tiles.back().ix!=x || p.tiles.back().iy!=y)
            p.tiles.push_back({static_cast<int>(n),0,x,y});
        ++p.tiles.back().count;
        p.z_index.push_back(p.axes.offset[2]+modes[3*n+2]);
    }
    return p;
}
// Positions are Cartesian and cell vectors are rows. One sin/cos pair gives all
// positive and negative integer harmonics along one reciprocal primitive vector.
POREMC_TENSOR_HD inline void harmonics(const Axes& axes, int d, const double* xyz, Complex* powers) {
    double f=xyz[0]*axes.inverse[d]+xyz[1]*axes.inverse[3+d]+xyz[2]*axes.inverse[6+d];
    f-=floor(f);
    constexpr double two_pi=6.283185307179586476925286766559;
    Complex unit;
#ifdef __CUDA_ARCH__
    sincos(two_pi*f,&unit.im,&unit.re);
#else
    unit={std::cos(two_pi*f),std::sin(two_pi*f)};
#endif
    Complex v{1.0,0.0};
    powers[axes.offset[d]]=v;
    for (int j=1;j<=axes.limit[d];++j) {
        v=multiply(v,unit);
        powers[axes.offset[d]+j]=v;
        powers[axes.offset[d]-j]={v.re,-v.im};
    }
}
POREMC_TENSOR_HD inline double energy(double weight, double base_re, double base_im, Complex s) {
    return weight*(2*(base_re*s.re+base_im*s.im)+s.re*s.re+s.im*s.im);
}
}
#undef POREMC_TENSOR_HD
#endif
