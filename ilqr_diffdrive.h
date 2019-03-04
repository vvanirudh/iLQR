#include "ilqr.h"

// Set dimensions
const int X_DIM = 3;
const int U_DIM = 2;
const int DIM = 2;

struct Obstacle {
    // spherical obstacle
    Vector<DIM> pos;
    double radius;
    int dim;
};

struct DiffDriveEnv {
    // obstacles
    std::vector<Obstacle> obstacles;
    double obstacleFactor;
    double scaleFactor;    
    // robot
    double robotRadius;
    double rotCost;
    // other attributes
    SymmetricMatrix<X_DIM> Q;
    Vector<X_DIM> xGoal, xStart;
    SymmetricMatrix<U_DIM> R;
    Vector<U_DIM> uNominal;
    Vector<DIM> bottomLeft, topRight;
    double dt;
    int ell;

    double obstacleCost(const Vector<X_DIM>& x);
    void quadratizeObstacleCost(const Vector<X_DIM>& x, SymmetricMatrix<X_DIM>& Q, Vector<X_DIM>& q);

    // Continuous-time dynamics \dot{x} = f(x, u)
    Vector<X_DIM> f(const Vector<X_DIM>& x, const Vector<U_DIM>& u);

    void regularize(SymmetricMatrix<DIM>& Q);
};

// local cost function c_t(x_t, u_t)
double ct(void* env, const Vector<X_DIM>& x, const Vector<U_DIM>& u, const int& t);
void quadratizeCost(void* env,
                    const Vector<X_DIM>& x,
                    const Vector<U_DIM>& u,
                    const int& t,
                    Matrix<U_DIM,X_DIM>& Pt,
                    SymmetricMatrix<X_DIM>& Qt,
                    SymmetricMatrix<U_DIM>& Rt,
                    Vector<X_DIM>& qt,
                    Vector<U_DIM>& rt,
                    const int& iter);

// final cost function
double cell(void* env, const Vector<X_DIM>& x);
void quadratizeFinalCost(void* env, const Vector<X_DIM>& x, SymmetricMatrix<X_DIM>& Qell, Vector<X_DIM>& qell, const int& iter);

// Discrete-time dynamics x_{t+1} = g(x_t, u_t)
Vector<X_DIM> g(void* env, const Vector<X_DIM>& x, const Vector<U_DIM>& u);
