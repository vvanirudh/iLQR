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

    env.R = 0.6*Matrix<U_DIM, U_DIM>::Identity();

    env.uNominal[0] = 2.5; 
    env.uNominal[1] = 2.5; 

    env.obstacleFactor = 1;
    env.scaleFactor = 1;

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

    int numIter;

    clock_t beginTime = clock();

    Vector<U_DIM> uNominal = Vector<U_DIM>::Zero();

    void* g_env = &env;
    iterativeLQR(env.ell, env.xStart, uNominal, g, quadratizeFinalCost, cell, quadratizeCost, ct, L, l, true, numIter, g_env);

    clock_t endTime = clock();
    std::cout << "Iterative LQR: NumIter: " << numIter << " Time: " << (endTime - beginTime) / (double) CLOCKS_PER_SEC << std::endl;

    // Execute control policy
    Vector<X_DIM> x = env.xStart;
    for (int t = 0; t < env.ell; ++t) {
        std::cout << t << ": " << x.transpose() << std::endl;
        x = g(&env, x, L[t]*x + l[t]);
    }
    std::cout << env.ell << ": " << x.transpose() << std::endl;

    return 0;
}
