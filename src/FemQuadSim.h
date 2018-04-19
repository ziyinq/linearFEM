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
    using TH = Eigen::Matrix<T, 6, dim>;
    using TD = Eigen::Matrix<T, dim, 6>;
    using TN = Eigen::Matrix<T, 6, 3>;
    using TMassN = Eigen::Matrix<T, 6, 1>;

    FemQuadSim(T dt, T E): dt(dt), E(E) {}
    ~FemQuadSim() {}

    void createMesh();
    void createOneMesh();
    void initialize();
    void startSimulation();
    void buildForce(int c);
    void advection(int c);
    void writeFrame(int framNum);
    Eigen::Matrix<T, 6, 1> computeN(T x, T y);
    Eigen::Matrix<T, 6, dim> computeH(T x, T y);
    TM linearPiola(TM F);
    TM neohookeanPiola(TM F);

  private:
    std::vector<Eigen::Matrix<int, 6, 1>> mesh;
    std::vector<TV> positions;
    std::vector<TV> velocities;
    std::vector<int> boundaryIdx;

    TH H1, H2, H3;
    std::vector<TH> massH;
    std::vector<TMassN> massN;
    std::vector<TM> DmHInv;
    std::vector<T> DmHDet;
    Eigen::MatrixXf massM;
    Eigen::MatrixXf forceVec;

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