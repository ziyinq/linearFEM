#include <Partio.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <fstream>

template<class T, int dim>
class FemSimulation{
public:
    using TV = Eigen::Matrix<T,dim,1>;
    using TM = Eigen::Matrix<T,dim,dim>;

    FemSimulation(T dt, T E): dt(dt), E(E) {}
    ~FemSimulation() {}

    void createMesh();
    void initialize();
    void startSimulation();
    void buildForce(int c);
    void advection(int c);
    void writeFrame(int framNum);
    TM linearPiola(TM F);
    TM neohookeanPiola(TM F);

  private:
    std::vector<Eigen::Matrix<int, dim + 1, 1>> mesh;
    std::vector<TV> positions;
    std::vector<TV> velocities;
    std::vector<int> boundaryIdx;

    Eigen::MatrixXf massM;
    Eigen::MatrixXf forceVec;
    std::vector<TM> DmInv;
    std::vector<T> W;
    std::vector<TV> force;

    int nodeNum;
    T density = 100;
    T width = 0.9;
    T height = 0.3;
    TV gravity = TV(0, -9.8);

    // simulation settings
    T dt;
    int numSteps = 300;

    // Young's modulus and Poisson's ratio
    T E;
    T nu = 0.3;
    T mu, lambda;

    // Cholesky solver
    Eigen::LDLT<Eigen::MatrixXf> ldlt;
};