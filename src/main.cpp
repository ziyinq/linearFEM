#include <iostream>
#include "FemSimulation.h"
#include "FemQuadSim.h"

using T = float;
constexpr int dim = 2;

int main()
{
    // FemSimulation<T, dim> sim;
    // sim.createMesh();
    T dt = 1e-3;
    T Youngs = 1.5e4;

    FemQuadSim<T, dim> quadSim(dt, Youngs);
    quadSim.startSimulation();

    FemSimulation<T, dim> linearSim(dt, Youngs);
    linearSim.startSimulation();

}