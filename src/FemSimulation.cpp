#include "FemSimulation.h"

template<class T, int dim>
void FemSimulation<T,dim>::initialize(){
    
}

template<class T, int dim>
void FemSimulation<T,dim>::createMesh(){
    int xdim = 10;
    int ydim = 4;
    T dx = 1.f / xdim;
    T dy = 0.4f / ydim;

    // generate points for mesh
    for(int i = 0; i < xdim; i++){
        for (int j = 0; j < ydim; j++){
            Eigen::Matrix<T,dim,1> point(i*dx, j*dy);
            positions.push_back(point);
        }
    }

    // generate index for mesh
    for (int i = 0; i < xdim-1; i++)
    {
        for (int j = 0; j < ydim-1; j++)
        {
            int a = i*ydim + j;
            int b = i*ydim + j+1;
            int c = (i+1)*ydim + j+1;
            int d = (i+1)*ydim + j;
            Eigen::Matrix<int, dim+1, 1> first(a, b, d);
            Eigen::Matrix<int, dim+1, 1> second(b, c, d);
            mesh.push_back(first);
            mesh.push_back(second);
        }
    }
}

template<class T, int dim>
void FemSimulation<T,dim>::writeFile(){
    Partio::ParticlesDataMutable *parts = Partio::create();
    Partio::ParticleAttribute posH;
    // mH = parts->addAttribute("m", Partio::VECTOR, 1);
    posH = parts->addAttribute("position", Partio::VECTOR, dim);
    // vH = parts->addAttribute("v", Partio::VECTOR, 3);

    for (unsigned int i = 0; i < positions.size(); i++)
    {
        int idx = parts->addParticle();
        float *p = parts->dataWrite<float>(posH, idx);
        for (int k = 0; k < dim; k++)
            p[k] = positions[i](k);
    }

    std::string particleFile = "../output/frame" + std::to_string(0) + ".bgeo";
    Partio::write(particleFile.c_str(), *parts);
    parts->release();

    // write .obj file
    std::ofstream fs;
    std::string objFile = "../output/mesh.obj";
    fs.open(objFile);
    for (auto X : positions)
    {
        fs << "v";
        for (int i = 0; i < dim; i++)
            fs << " " << X(i);
        if (dim == 2)
            fs << " 0";
        fs << "\n";
    }
    for (auto F : mesh)
    {
        fs << "f";
        for (int i = 0; i < dim+1; i++)
            fs << " " << F(i) + 1;
        fs << "\n";
    }
    fs.close();
}

template class FemSimulation<float, 2>;