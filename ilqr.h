#ifndef __ILQR_H__
#define __ILQR_H__

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <vector>
#include <math.h>
#include <iostream>
#include <limits>

template <int Dim>
using Vector = Eigen::Matrix<double, Dim, 1>;

template <int Dim>
using SymmetricMatrix = Eigen::Matrix<double, Dim, Dim>;

template <int rDim, int cDim>
using Matrix = Eigen::Matrix<double, rDim, cDim>;

static const double DEFAULTSTEPSIZE = 0.0009765625;

template <int aDim, typename T, int yDim>
    inline Matrix<yDim,aDim> dyn_jacobian_x(const Vector<aDim>& a, const T& b, Vector<yDim> (*f)(void*, const Vector<aDim>&, const T&), void* env, double jStep=DEFAULTSTEPSIZE) {
    /*
      Computes the jacobian of a function using finite differences w.r.t first component

      a - first component
      b - second component
      f - function f(a, b) whose jacobian is to be computed
     */    
    Matrix<yDim,aDim> A;
    Vector<aDim> ar(a), al(a);
    for (int i = 0; i < aDim; ++i) {
        ar[i] += jStep; al[i] -= jStep;
        //A.insert(0,i, (f(ar, b) - f(al, b)) / (2*jStep));
        // assign column
        A.col(i) = (f(env, ar, b) - f(env, al, b)) / (2*jStep);
        ar[i] = al[i] = a[i];
    }
    return A;
}

template <typename T1, int bDim, int yDim>
    inline Matrix<yDim,bDim> dyn_jacobian_u(const T1& a, const Vector<bDim>& b, Vector<yDim> (*f)(void*, const T1&, const Vector<bDim>&), void* env, double jStep=DEFAULTSTEPSIZE) {
    /*
      Computes the jacobian of a function using finite differences w.r.t second component

      a - first component
      b - second component
      f - function f(a, b) whose jacobian is to be computed
    */
    Matrix<yDim,bDim> B;
    Vector<bDim> br(b), bl(b);
    for (int i = 0; i < bDim; ++i) {
        br[i] += jStep; bl[i] -= jStep;
        //B.insert(0,i, (f(a, br) - f(a, bl)) / (2*jStep));
        // assign column
        B.col(i) = (f(env, a, br) - f(env, a, bl)) / (2*jStep);
        br[i] = bl[i] = b[i];
    }
    return B;
}

template <int xDim, int uDim>
inline void iterativeLQR(const int& ell,
                  const Vector<xDim>& initState,
                  const Vector<uDim>& uNominal,
                  Vector<xDim> (*g)(void*, const Vector<xDim>&, const Vector<uDim>&),
                  void (*quadratizeFinalCost)(void*, const Vector<xDim>&, SymmetricMatrix<xDim>&, Vector<xDim>&, const int&),
                  double (*cell)(void*, const Vector<xDim>&),
                  void (*quadratizeCost)(void*, const Vector<xDim>&, const Vector<uDim>&, const int&, Matrix<uDim, xDim>&, SymmetricMatrix<xDim>&, SymmetricMatrix<uDim>&, Vector<xDim>&, Vector<uDim>&, const int&),
                  double (*ct)(void*, const Vector<xDim>&, const Vector<uDim>&, const int&),
                  std::vector<Matrix<uDim, xDim> >& L,
                  std::vector<Vector<uDim> >&l,
                  bool vis,
                  int& iter,
                  void* env) {
    /*
      ell - Horizon length (or T, in notes)
      initState - initial state (or x_0, in notes)
      uNominal - Nominal control input
      g - dynamics function (or f, in notes)
      quadratizeFinalCost - Given the final state, quadratize the final cost
      cell - Final state cost function (true cost function? TODO:)
      quadratizeCost - Given state and control, quadratize cost
      ct - Cost function (true cost function? TODO:)
      L - TODO:
      l - resulting control input (TODO:)
      vis - verbose
      iter - iteration number


      Runs iLQR for a maximum of 1000 iterations or returns if there's no significant improvement in the cost across iterations
     */

    int maxIter = 1000;

    L.resize(ell, Matrix<uDim, xDim>::Zero());
    l.resize(ell, uNominal);

    std::vector<Vector<xDim> > xHat(ell + 1, Vector<xDim>::Zero());
    std::vector<Vector<xDim> > xHatNew(ell + 1, Vector<xDim>::Zero());
    std::vector<Vector<uDim> > uHat(ell);
    std::vector<Vector<uDim> > uHatNew(ell);

    double oldCost = -log(0.0);

    for (iter = 0; iter < maxIter; ++iter) {
        double newCost;
        double alpha = 1.0;

        // Forward pass to get nominal trajectory
        do {
            newCost = 0;

            // initialize trajectory
            xHatNew[0] = initState;
            for (int t = 0; t < ell; ++t) {
                // Compute control
                uHatNew[t] = (1.0 - alpha)*uHat[t] + L[t]*(xHatNew[t] - (1.0 - alpha)*xHat[t]) + alpha*l[t];
                //std::cout<<"u : "<<uHatNew[t] << std::endl;
                // Forward one-step
                xHatNew[t+1] = g(env, xHatNew[t], uHatNew[t]);
                //std::cout<<"newx : "<<xHatNew[t+1] << std::endl;
                
                // compute cost
                newCost += ct(env, xHatNew[t], uHatNew[t], t);
                //std::cout<<"cost : " <<newCost << std::endl;
            }
            // Compute final state cost
            newCost += cell(env, xHatNew[ell]);
            //std::cout<<"final cost : " << newCost << std::endl;
            //exit(0);

            // Decrease alpha, if the new cost is not less than old cost
            alpha *= 0.5;
            //std::cout << "Old cost : "<< oldCost << " New cost : " << newCost << std::endl;
        } while (!(newCost < oldCost || fabs((oldCost - newCost) / newCost) < 1e-4));

        xHat = xHatNew;
        uHat = uHatNew;

        if (vis) {
            std::cout << "Iter: " << iter << " Alpha: " << 2*alpha << " Rel. progress: " << (oldCost - newCost) / newCost << " Cost: " << newCost << " Time step: " << exp(xHat[0][xDim-1]) << std::endl;
        }

        if (fabs((oldCost - newCost) / newCost) < 1e-4) {
            // No significant improvement in cost
            //std::cout << "returned with value " << fabs((oldCost - newCost) / newCost) << std::endl;
            return;
        }

        oldCost = newCost;

        // backward pass to compute updates to control
        SymmetricMatrix<xDim> S;
        Vector<xDim> s;  // v, in notes

        // compute final cost  S_N = Q_f
        quadratizeFinalCost(env, xHat[ell], S, s, iter);

        for (int t = ell-1; t != -1; --t) {
            // Compute A_t and B_t (derivatives of dynamics w.r.t x and u)
            const SymmetricMatrix<xDim> A = dyn_jacobian_x(xHat[t], uHat[t], g, env);
            const Matrix<xDim, uDim> B = dyn_jacobian_u(xHat[t], uHat[t], g, env);
            const Vector<xDim> c = xHat[t+1] - A*xHat[t] - B*uHat[t];  // error in linearization

            Matrix<uDim, xDim> P;
            SymmetricMatrix<xDim> Q;
            SymmetricMatrix<uDim> R;
            Vector<xDim> q;
            Vector<uDim> r;

            // Quadratize the cost
            quadratizeCost(env, xHat[t], uHat[t], t, P, Q, R, q, r, iter);

            const Matrix<uDim, xDim> C = B.transpose()*S*A + P;
            const SymmetricMatrix<xDim> D = A.transpose()*(S*A) + Q;
            const SymmetricMatrix<uDim> E = B.transpose()*(S*B) + R;
            const Vector<xDim> d = A.transpose()*(s + S*c) + q;
            const Vector<uDim> e = B.transpose()*(s + S*c) + r;

            L[t] = -(E.colPivHouseholderQr().solve(C));
            l[t] = -(E.colPivHouseholderQr().solve(e));

            S = D + C.transpose()*L[t];
            s = d + C.transpose()*l[t];
        }
        
    }
}


#endif
