#include "ilqr.h"
#include <limits>
#include <iostream>

template <size_t xDim, size_t uDim>
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
                  size_t& iter) {
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

    size_t maxIter = 1000;

    L.resize(ell, Matrix<uDim, xDim>::Zero());
    l.resize(ell, uNominal);

    std::vector<Vector<xDim> > xHat(ell + 1, Vector<xDim>::Zero());
    std::vector<Vector<xDim> > xHatNew(ell + 1, Vector<xDim>::Zero());
    std::vector<Vector<uDim> > uHat(ell);
    std::vector<Vector<uDim> > uHatNew(ell);

    double oldCost = std::numeric_limits<double>::max();

    for (iter = 0; iter < maxIter; ++iter) {
        double newCost;
        double alpha = 0;

        // Forward pass to get nominal trajectory
        do {
            newCost = 0;

            // initialize trajectory
            xHatNew[0] = initState;
            for (size_t t = 0; t < ell; ++t) {
                // Compute control
                uHatNew[t] = (1.0 - alpha)*uHat[t] + L[t]*(xHatNew[t] - (1.0 - alpha)*xHat[t]) + alpha*l[t];
                // Forward one-step
                xHatNew[t+1] = g(xHatNew[t], uHatNew[t]);
                
                // compute cost
                newCost += ct(xHatNew[t], uHatNew[t], t);
            }
            // Compute final state cost
            newCost += cell(xHatNew[ell]);

            // Decrease alpha, if the new cost is not less than old cost
            alpha *= 0.5;
        } while (!(newCost < oldCost || abs((oldCost - newCost) / newCost) < 1e-4));

        xHat = xHatNew;
        uHat = uHatNew;

        if (vis) {
            std::cout << "Iter: " << iter << " Alpha: " << 2*alpha << " Rel. progress: " << (oldCost - newCost) / newCost << " Cost: " << newCost << " Time step: " << exp(xHat[0][xDim-1]) << std::endl;
        }

        if (abs((oldCost - newCost) / newCost) < 1e-4) {
            // No significant improvement in cost
            return;
        }

        oldCost = newCost;

        // backward pass to compute updates to control
        SymmetricMatrix<xDim> S;
        Vector<xDim> s;  // v, in notes

        // compute final cost  S_N = Q_f
        quadratizeFinalCost(xHat[ell], S, s, iter);

        for (size_t t = ell-1; t != -1; --t) {
            // Compute A_t and B_t (derivatives of dynamics w.r.t x and u)
            const SymmetricMatrix<xDim> A = dyn_jacobian_x(xHat[t], uHat[t], g);
            const Matrix<xDim, uDim> B = dyn_jacobian_u(xHat[t], uHat[t], g);
            const Vector<xDim> c = xHat[t+1] - A*xHat[t] - B*uHat[t];  // error in linearization

            Matrix<uDim, xDim> P;
            SymmetricMatrix<xDim> Q;
            SymmetricMatrix<uDim> R;
            Vector<xDim> q;
            Vector<uDim> r;

            // Quadratize the cost
            quadratizeCost(xHat[t], uHat[t], t, P, Q, R, q, r, iter);

            const Matrix<uDim, xDim> C = B.transpose()*S*A + P;
            const SymmetricMatrix<xDim> D = A.transpose()*(S*A) + Q;
            const SymmetricMatrix<uDim> E = B.transpose()*(S*B) + R;
            const Vector<xDim> d = A.transpose()*(s + S*c) + q;
            const Vector<uDim> e = B.tranpose()*(s + S*c) + r;

            L[t] = -(E.colPivHouseholderQr().solve(C));
            l[t] = -(E.colPivHouseholderQr().solve(e));

            S = D + C.transpose()*L[t];
            s = d + C.tranpose()*l[t];
        }
        
    }
}

template <size_t aDim, typename T, size_t yDim>
Matrix<yDim,aDim> dyn_jacobian_x(const Vector<aDim>& a, const T& b, Vector<yDim> (*f)(const Vector<aDim>&, const T&), double jStep) {
    /*
      Computes the jacobian of a function using finite differences w.r.t first component

      a - first component
      b - second component
      f - function f(a, b) whose jacobian is to be computed
     */
    Matrix<yDim,aDim> A;
    Vector<aDim> ar(a), al(a);
    for (size_t i = 0; i < aDim; ++i) {
        ar[i] += jStep; al[i] -= jStep;
        //A.insert(0,i, (f(ar, b) - f(al, b)) / (2*jStep));
        // assign column
        A.col(i) = (f(ar, b) - f(al, b)) / (2*jStep);
        ar[i] = al[i] = a[i];
    }
    return A;
}

template <typename T1, size_t bDim, size_t yDim>
Matrix<yDim,bDim> dyn_jacobian_u(const T1& a, const Vector<bDim>& b, Vector<yDim> (*f)(const T1&, const Vector<bDim>&), double jStep) {
    /*
      Computes the jacobian of a function using finite differences w.r.t second component

      a - first component
      b - second component
      f - function f(a, b) whose jacobian is to be computed
    */
    Matrix<yDim,bDim> B;
    Vector<bDim> br(b), bl(b);
    for (size_t i = 0; i < bDim; ++i) {
        br[i] += jStep; bl[i] -= jStep;
        //B.insert(0,i, (f(a, br) - f(a, bl)) / (2*jStep));
        // assign column
        B.col(i) = (f(a, br) - f(a, bl)) / (2*jStep);
        br[i] = bl[i] = b[i];
    }
    return B;
}
