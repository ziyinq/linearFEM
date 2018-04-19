#include <iostream>
#include "FemSimulation.h"
#include "FemQuadSim.h"

using T = float;
constexpr int dim = 2;

int main()
{
    // FemSimulation<T, dim> sim;
    // sim.createMesh();
    FemQuadSim<T, dim> quadSim;
    FemSimulation<T, dim> linearSim;
    // linearSim.startSimulation();
    linearSim.startSimulation();
}