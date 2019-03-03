#include "ilqr.h"
#include "ilqr_diffdrive.h"
#include <iostream>

int main() {

    DiffDriveEnv env;

    // control parameters
    env.ell = 150;
    env.dt = 1.0/6.0;
    
    env.Q = 50*Matrix<X_DIM, X_DIM>::Identity();
    env.rotCost = 0.4;

    env.xGoal = Vector<X_DIM>::Zero();
    env.xGoal[0] = 0;
    env.xGoal[1] = 25;
    env.xGoal[2] = M_PI; 

    env.xStart = Vector<X_DIM>::Zero();
    env.xStart[0] = 0;
    env.xStart[1] = -25;
    env.xStart[2] = M_PI;

    env.R = 0.6*Matrix<U_DIM, U_DIM>::Zero();

    env.uNominal[0] = 2.5; 
    env.uNominal[1] = 2.5; 

    env.obstacleFactor = 1.0;
    env.scaleFactor = 1.0;

    // Environment settings
    env.robotRadius = 3.35/2.0;
    env.bottomLeft[0] = -20; env.bottomLeft[1] = -30;
    env.topRight[0] = 20; env.topRight[1] = 30;

    Obstacle obstacle;
    obstacle.pos[0] = 0; obstacle.pos[1] = -13.5;
    obstacle.radius = 2.0;
    env.obstacles.push_back(obstacle);

    obstacle.pos[0] = 10; obstacle.pos[1] = -5;
    obstacle.radius = 2.0;
    env.obstacles.push_back(obstacle);

    obstacle.pos[0] = -9.5; obstacle.pos[1] = -5;
    obstacle.radius = 2.0;
    env.obstacles.push_back(obstacle);

    obstacle.pos[0] = -2; obstacle.pos[1] = 3;
    obstacle.radius = 2.0;
    env.obstacles.push_back(obstacle);

    obstacle.pos[0] = 8; obstacle.pos[1] = 7;
    obstacle.radius = 2.0;
    env.obstacles.push_back(obstacle);

    obstacle.pos[0] = 11; obstacle.pos[1] = 20;
    obstacle.radius = 2.0;
    env.obstacles.push_back(obstacle);

    obstacle.pos[0] = -12; obstacle.pos[1] = 8;
    obstacle.radius = 2.0;
    env.obstacles.push_back(obstacle);

    obstacle.pos[0] = -11; obstacle.pos[1] = 21;
    obstacle.radius = 2.0;
    env.obstacles.push_back(obstacle);

    obstacle.pos[0] = -1; obstacle.pos[1] = 16;
    obstacle.radius = 2.0;
    env.obstacles.push_back(obstacle);

    obstacle.pos[0] = -11; obstacle.pos[1] = -19;
    obstacle.radius = 2.0;
    env.obstacles.push_back(obstacle);

    obstacle.pos[0] = 10 + sqrt(2.0); obstacle.pos[1] = -15 - sqrt(2.0);
    obstacle.radius = 2.0;
    env.obstacles.push_back(obstacle);

    std::vector<Matrix<U_DIM, X_DIM> > L;
    std::vector<Vector<U_DIM> > l;

    size_t numIter;

    clock_t beginTime = clock();

    std::function<Vector<X_DIM>(const Vector<X_DIM>&, const Vector<U_DIM>&)> gfp = std::bind(&DiffDriveEnv::g,
                                                                                             &env,
                                                                                             std::placeholders::_1,
                                                                                             std::placeholders::_2);
    std::function<void(const Vector<X_DIM>&, SymmetricMatrix<X_DIM>&, Vector<X_DIM>&, const size_t&)> quadratizeFinalCostfp = std::bind(&DiffDriveEnv::quadratizeFinalCost,
                                                                                                                                        &env,
                                                                                                                                        std::placeholders::_1,
                                                                                                                                        std::placeholders::_2,
                                                                                                                                        std::placeholders::_3,
                                                                                                                                        std::placeholders::_4);
    std::function<double(const Vector<X_DIM>&)> cellfp = std::bind(&DiffDriveEnv::cell,
                                                                   &env,
                                                                   std::placeholders::_1);
    std::function<void(const Vector<X_DIM>&, const Vector<U_DIM>&, const size_t&, Matrix<U_DIM, X_DIM>&, SymmetricMatrix<X_DIM>&, SymmetricMatrix<U_DIM>&, Vector<X_DIM>&, Vector<U_DIM>&, const size_t&)> quadratizeCostfp = std::bind(&DiffDriveEnv::quadratizeCost, &env,
                                      std::placeholders::_1,
                                      std::placeholders::_2,
                                      std::placeholders::_3,
                                      std::placeholders::_4,
                                      std::placeholders::_5,
                                      std::placeholders::_6,
                                      std::placeholders::_7,
                                      std::placeholders::_8,
                                      std::placeholders::_9);
    std::function<double(const Vector<X_DIM>&, const Vector<U_DIM>&, const size_t&)> ctfp = std::bind(&DiffDriveEnv::ct,
                                                                                                      &env,
                                                                                                      std::placeholders::_1,
                                                                                                      std::placeholders::_2,
                                                                                                      std::placeholders::_3);

    Vector<U_DIM> uNominal = Vector<U_DIM>::Zero();
    iterativeLQR(env.ell, env.xStart, uNominal, gfp, quadratizeFinalCostfp, cellfp, quadratizeCostfp, ctfp, L, l, true, numIter);

    clock_t endTime = clock();
    std::cerr << "Iterative LQR: NumIter: " << numIter << " Time: " << (endTime - beginTime) / (double) CLOCKS_PER_SEC << std::endl;

    // Execute control policy
    Vector<X_DIM> x = env.xStart;
    for (size_t t = 0; t < env.ell; ++t) {
        std::cerr << t << ": " << x.transpose();
        x = env.g(x, L[t]*x + l[t]);
    }
    std::cerr << env.ell << ": " << x.transpose();

    return 0;
}
