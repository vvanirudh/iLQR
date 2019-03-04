#include "ilqr_diffdrive.h"

double DiffDriveEnv::obstacleCost(const Vector<X_DIM>& x) {
    double cost = 0;

    for (int i=0; i < this->obstacles.size(); ++i) {
        Vector<DIM> d = x.head(DIM) - this->obstacles[i].pos;
        double distr = sqrt(d.dot(d.transpose()));
        double dist = distr - this->robotRadius - this->obstacles[i].radius;
        // TODO: This needs to be changed to hinge cost
        cost += this->obstacleFactor * exp(-this->scaleFactor * dist);
    }

    for (int i=0; i < DIM; ++i) {
        double dist = (x[i] - this->bottomLeft[i]) - this->robotRadius;
        cost += this->obstacleFactor * exp(-this->scaleFactor * dist);
    }

    for (int i=0; i<DIM; ++i) {
        double dist = (this->topRight[i] - x[i]) - this->robotRadius;
        cost += this->obstacleFactor * exp(-this->scaleFactor * dist);
    }

    return cost;
}

void DiffDriveEnv::quadratizeObstacleCost(const Vector<X_DIM>& x, SymmetricMatrix<X_DIM>& Q, Vector<X_DIM>& q) {
    SymmetricMatrix<DIM> QObs = SymmetricMatrix<DIM>::Zero();
    Vector<DIM> qObs = Vector<DIM>::Zero();

    for (int i = 0; i < this->obstacles.size(); ++i) {
        Vector<DIM> d = x.head(DIM) - this->obstacles[i].pos;
        double distr = sqrt(d.dot(d.transpose()));
        d /= distr;
        double dist = distr - this->robotRadius - this->obstacles[i].radius;
        
        Vector<DIM> d_ortho;
        d_ortho[0] = d[1];
        d_ortho[1] = -d[0];
        
        double a0 = this->obstacleFactor * exp(-this->scaleFactor*dist);
        double a1 = -this->scaleFactor*a0;
        double a2 = -this->scaleFactor*a1;
        
        double b2 = a1 / distr;
        
        QObs += a2*(d*d.transpose()) + b2*(d_ortho*d_ortho.transpose());
        qObs += a1*d;
    }
    for (int i = 0; i < DIM; ++i) {
        double dist = (x[i] - this->bottomLeft[i]) - this->robotRadius;
        
        Vector<DIM> d = Vector<DIM>::Zero();
        d[i] = 1.0;
        
        double a0 = this->obstacleFactor * exp(-this->scaleFactor*dist);
        double a1 = -this->scaleFactor*a0;
        double a2 = -this->scaleFactor*a1;
        
        QObs += a2*(d*d.transpose());
        qObs += a1*d;
    }
    for (int i = 0; i < DIM; ++i) {
        double dist = (this->topRight[i] - x[i]) - this->robotRadius;
        
        Vector<DIM> d = Vector<DIM>::Zero();
        d[i] = -1.0;
        
        double a0 = this->obstacleFactor * exp(-this->scaleFactor*dist);
        double a1 = -this->scaleFactor*a0;
        double a2 = -this->scaleFactor*a1;
        
        QObs += a2*(d*d.transpose());
        qObs += a1*d;
    }
    
    this->regularize(QObs);
    //Q.insert(0, QObs + Q.subSymmetricMatrix<DIM>(0));
    Q.block<DIM, DIM>(0, 0) = QObs + Q.block<DIM, DIM>(0, 0);
    //q.insert(0,0, qObs - QObs*x.subMatrix<DIM>(0,0) + q.subMatrix<DIM>(0,0));
    q.head(DIM) = qObs - QObs * x.head(DIM) + q.head(DIM);
}

double ct(void* env, const Vector<X_DIM>& x, const Vector<U_DIM>& u, const int& t) {
    DiffDriveEnv* g_env = static_cast<DiffDriveEnv*>(env);
    double cost = 0;
    if (t == 0) {
        cost += ((x - g_env->xStart).transpose()*g_env->Q*(x - g_env->xStart));
    }
    cost += ((u - g_env->uNominal).transpose()*g_env->R*(u - g_env->uNominal));
    cost += g_env->obstacleCost(x);
    return cost;
}

void quadratizeCost(void* env, const Vector<X_DIM>& x, const Vector<U_DIM>& u, const int& t, Matrix<U_DIM,X_DIM>& Pt, SymmetricMatrix<X_DIM>& Qt, SymmetricMatrix<U_DIM>& Rt, Vector<X_DIM>& qt, Vector<U_DIM>& rt, const int& iter) {
    /*Qt = hessian1(x, u, t, c); 
      Pt = ~hessian12(x, u, t, c);
      Rt = hessian2(x, u, t, c);
      qt = jacobian1(x, u, t, c) - Qt*x - ~Pt*u;
      rt = jacobian2(x, u, t, c) - Pt*x - Rt*u;*/
    DiffDriveEnv* g_env = static_cast<DiffDriveEnv*>(env);
    if (t == 0) {
        Qt = g_env->Q;
        qt = -(g_env->Q*g_env->xStart);
    } else {
        Qt = SymmetricMatrix<X_DIM>::Zero(); 
        qt = Vector<X_DIM>::Zero();
        
        if (iter < 2) {
            Qt(2,2) = g_env->rotCost;
            qt[2] = -g_env->rotCost*(M_PI/2);
        }
    }
    Rt = g_env->R;
    rt = -(g_env->R*g_env->uNominal);
    Pt = Matrix<U_DIM, X_DIM>::Zero();
    
    g_env->quadratizeObstacleCost(x, Qt, qt);
}

// Final cost function c_\ell(x_\ell)
double cell(void* env, const Vector<X_DIM>& x) {
    DiffDriveEnv* g_env = static_cast<DiffDriveEnv*>(env);
    double cost = 0;
    cost += ((x - g_env->xGoal).transpose()*g_env->Q*(x - g_env->xGoal));
    return cost;
}

void quadratizeFinalCost(void* env, const Vector<X_DIM>& x, SymmetricMatrix<X_DIM>& Qell, Vector<X_DIM>& qell, const int& iter) {
    DiffDriveEnv* g_env = static_cast<DiffDriveEnv*>(env);
    /*Qell = hessian(x, cell); 
      qell = jacobian(x, cell) - Qell*x;*/
    Qell = g_env->Q;
    qell = -(g_env->Q*g_env->xGoal);
}

// Continuous-time dynamics \dot{x} = f(x,u)
Vector<X_DIM> DiffDriveEnv::f(const Vector<X_DIM>& x, const Vector<U_DIM>& u) {
    Vector<X_DIM> xDot;

    // Differential-drive
    xDot[0] = 0.5*(u[0] + u[1])*cos(x[2]);
    xDot[1] = 0.5*(u[0] + u[1])*sin(x[2]);
    xDot[2] = (u[1] - u[0])/2.58;

    return xDot;
}

// Discrete-time dynamics x_{t+1} = g(x_t, u_t)
Vector<X_DIM> g(void* env, const Vector<X_DIM>& x, const Vector<U_DIM>& u) {
    DiffDriveEnv* g_env = static_cast<DiffDriveEnv*>(env);
    Vector<X_DIM> k1 = g_env->f(x, u);
    Vector<X_DIM> k2 = g_env->f(x + 0.5*g_env->dt*k1, u);
    Vector<X_DIM> k3 = g_env->f(x + 0.5*g_env->dt*k2, u);
    Vector<X_DIM> k4 = g_env->f(x + g_env->dt*k3, u);
    return x + (g_env->dt/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4);
}

void DiffDriveEnv::regularize(SymmetricMatrix<DIM>& Q) {
    SymmetricMatrix<DIM> D;
    Matrix<DIM, DIM> V;
    //jacobi(Q, V, D);
    Eigen::EigenSolver<SymmetricMatrix<DIM> > es(Q);
    D = es.pseudoEigenvalueMatrix();
    V = es.pseudoEigenvectors();
    for (int i = 0; i < DIM; ++i) {
        if (D(i,i) < 0) {
            D(i,i) = 0;
        }
    }
    Q = V*(D*V.transpose());
}
