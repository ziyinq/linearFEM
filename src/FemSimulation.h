#include <Partio.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <vector>

template<class T, int dim>
class FemSimulation{
public:
    std::vector<Eigen::Matrix<T,dim,1>> positions;
    std::vector<Eigen::Matrix<T,dim,1>> velocities;
    std::vector<T> mass;

    FemSimulation(){}
    ~FemSimulation(){}
    
    void initialize();
    
};