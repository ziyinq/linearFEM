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
    FemSimulation(){}
    ~FemSimulation(){}

    void initialize();
    void precomputation();
    void createMesh();
    void writeFile();
private:
    std::vector<Eigen::Matrix<int, dim + 1, 1>> mesh;
    std::vector<TV> positions;
    std::vector<TV> velocities;
    std::vector<T> mass;

    std::vector<TM> Bm;
    std::vector<T> W;
};