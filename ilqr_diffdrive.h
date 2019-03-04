#include "ilqr.h"

// Set dimensions
#define X_DIM 3
#define U_DIM 2
#define DIM 2

struct Obstacle {
    // spherical obstacle
    Vector<DIM> pos;
    double radius;
    size_t dim;
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
    size_t ell;

    double obstacleCost(const Vector<X_DIM>& x);
    void quadratizeObstacleCost(const Vector<X_DIM>& x, SymmetricMatrix<X_DIM>& Q, Vector<X_DIM>& q);

    // Continuous-time dynamics \dot{x} = f(x, u)
    Vector<X_DIM> f(const Vector<X_DIM>& x, const Vector<U_DIM>& u);

    void regularize(SymmetricMatrix<DIM>& Q);
};

// local cost function c_t(x_t, u_t)
double ct(DiffDriveEnv& env, const Vector<X_DIM>& x, const Vector<U_DIM>& u, const size_t& t);
void quadratizeCost(DiffDriveEnv& env,
                    const Vector<X_DIM>& x,
                    const Vector<U_DIM>& u,
                    const size_t& t,
                    Matrix<U_DIM,X_DIM>& Pt,
                    SymmetricMatrix<X_DIM>& Qt,
                    SymmetricMatrix<U_DIM>& Rt,
                    Vector<X_DIM>& qt,
                    Vector<U_DIM>& rt,
                    const size_t& iter);

// final cost function
double cell(DiffDriveEnv& env, const Vector<X_DIM>& x);
void quadratizeFinalCost(DiffDriveEnv& env, const Vector<X_DIM>& x, SymmetricMatrix<X_DIM>& Qell, Vector<X_DIM>& qell, const size_t& iter);

// Discrete-time dynamics x_{t+1} = g(x_t, u_t)
Vector<X_DIM> g(DiffDriveEnv& env, const Vector<X_DIM>& x, const Vector<U_DIM>& u);
