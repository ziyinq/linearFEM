#include <iostream>
#include "FemSimulation.h"

using T = float;
constexpr int dim = 2;

int main()
{
    FemSimulation<T, dim> sim;
    sim.createMesh();
    sim.writeFile();
    std::cout << "main!" << std::endl;
}