#ifndef __ILQR_H__
#define __ILQR_H__

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <vector>
#include <math.h>
#include <functional>

template <size_t Dim>
using Vector = Eigen::Matrix<double, Dim, 1>;

template <size_t Dim>
using SymmetricMatrix = Eigen::Matrix<double, Dim, Dim>;

template <size_t rDim, size_t cDim>
using Matrix = Eigen::Matrix<double, rDim, cDim>;

static const double DEFAULTSTEPSIZE = 0.0009765625;

template <int xDim, int uDim>
void iterativeLQR(const size_t& ell,
                  const Vector<xDim>& initState,
                  const Vector<uDim>& uNominal,
                  Vector<xDim> (*g)(const Vector<xDim>&, const Vector<uDim>&),
                  void (*quadratizeFinalCost)(const Vector<xDim>&, SymmetricMatrix<xDim>&, Vector<xDim>&, const size_t&),
                  double (*cell)(const Vector<xDim>&),
                  void (*quadratizeCost)(const Vector<xDim>&, const Vector<uDim>&, const size_t&, Matrix<uDim, xDim>&, SymmetricMatrix<xDim>&, SymmetricMatrix<uDim>&, Vector<xDim>&, Vector<uDim>&, const size_t&),
                  double (*ct)(const Vector<xDim>&, const Vector<uDim>&, const size_t&),
                  std::vector<Matrix<uDim, xDim> >& L,
                  std::vector<Vector<uDim> >&l,
                  bool vis,
                  size_t& iter);

template <size_t aDim, typename T, size_t yDim>
Matrix<yDim,aDim> dyn_jacobian_x(const Vector<aDim>& a, const T& b, Vector<yDim> (*f)(const Vector<aDim>&, const T&), double jStep = DEFAULTSTEPSIZE);

template <typename T1, size_t bDim, size_t yDim>
Matrix<yDim,bDim> dyn_jacobian_u(const T1& a, const Vector<bDim>& b, Vector<yDim> (*f)(const T1&, const Vector<bDim>&), double jStep = DEFAULTSTEPSIZE);

#endif
