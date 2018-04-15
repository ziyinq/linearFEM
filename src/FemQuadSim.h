#include <Partio.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <fstream>

template <class T, int dim>
class FemQuadSim
{
  public:
    using TV = Eigen::Matrix<T, dim, 1>;
    using TM = Eigen::Matrix<T, dim, dim>;

    FemQuadSim() {}
    ~FemQuadSim() {}

    void createMesh();
    void initialize();
    void startSimulation();
    void advection();
    void writeFrame(int framNum);
    TM linearPiola(TM F);

  private:
    std::vector<Eigen::Matrix<int, 6, 1>> mesh;
    std::vector<TV> positions;
    std::vector<TV> velocities;
    std::vector<TV> force;
    std::vector<T> mass;
    std::vector<int> boundaryIdx;

    std::vector<TM> DmInv;
    std::vector<T> W;

    T density = 100;
    T width = 0.9;
    T height = 0.3;
    TV gravity = TV(0, -9.8);

    // simulation settings
    T dt = 5e-4;
    int numSteps = 300;

    // Young's modulus and Poisson's ratio
    T E = 1e4;
    T nu = 0.3;
    T mu, lambda;
};