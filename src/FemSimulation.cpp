#include "FemSimulation.h"

template<class T, int dim>
void FemSimulation<T,dim>::createMesh(){
    // TODO: not valid for 3D
    int xdim = 10;
    int ydim = 4;
    T dx = width / (xdim-1);
    T dy = height / (ydim-1);

    // generate points for mesh
    for(int i = 0; i < xdim; i++){
        for (int j = 0; j < ydim; j++){
            TV point(i*dx, j*dy);
            positions.push_back(point);
            // set left wall to be boundary wall
            if (i == 0) boundaryIdx.push_back(j);
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
            Eigen::Matrix<int, dim + 1, 1> first(a, b, d);
            Eigen::Matrix<int, dim + 1, 1> second(b, c, d);
            mesh.push_back(first);
            mesh.push_back(second);
        }
    }

    auto size = positions.size();
    mass.resize(size);
    velocities.resize(size);
    force.resize(size);

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
        for (int i = 0; i < dim + 1; i++)
            fs << " " << F(i) + 1;
        fs << "\n";
    }
    fs.close();
}

template<class T, int dim>
void FemSimulation<T,dim>::startSimulation(){
    std::cout << "======Simulation Starts!=====" << std::endl;
    createMesh();
    initialize();
    for (int step = 0; step < numSteps; step++){
        // reset force to be gravity
        std::fill(force.begin(), force.end(), TV::Zero());

        // calculate force
        for (size_t i = 0; i < mesh.size(); i++){
            TM Ds = TM::Zero();
            for (int j = 0; j < dim; j++){
                Ds.col(j) = positions[mesh[i](j)] - positions[mesh[i](dim)];
            }
            TM F = Ds*DmInv[i];
            TM P = linearPiola(F);
            TM H = -W[i]*P*DmInv[i].transpose();
            //std::cout << H << std::endl << std::endl;;
            TV lastForce = TV::Zero();
            for (int k = 0; k < dim; k++){
                force[mesh[i](k)] += H.col(k);
                lastForce += H.col(k);
            }
            force[mesh[i](dim)] -= lastForce;
        }

        // update velocity and advect node
        advection();
        writeFrame(step);
    }
}

template<class T, int dim>
void FemSimulation<T,dim>::advection(){
    for (size_t i = 0; i < positions.size(); i++){
        velocities[i] += dt * (force[i] / mass[i] + gravity);
    }

    // fix boundary velocity to be 0
    for (auto X : boundaryIdx){
        velocities[X] = TV::Zero();
    }

    for (size_t i = 0; i < positions.size(); i++)
    {
        positions[i] += dt * velocities[i];
    }
}

template <class T, int dim>
void FemSimulation<T, dim>::initialize()
{
    // initialize DmInv, W, velocities, mass 
    // valid for 2D and 3D
    for (auto X : mesh)
    {
        TM Dm = TM::Zero();
        for (int i = 0; i < dim; i++)
        {
            Dm.col(i) = positions[X(i)] - positions[X(dim)];
        }
        DmInv.push_back(Dm.inverse());
        T thisW = Dm.determinant();
        if (thisW < 0.f)
            thisW *= -1;
        W.push_back(thisW);

        // estimate mass
        // TODO: need modify if quadratic
        T thismass = thisW * density / 3;
        // TODO: fix if not linear FEM
        for (int i = 0; i < dim + 1; i++)
        {
            mass[X(i)] += thismass;
        }
    }

    // set velocity to be zero
    std::fill(velocities.begin(), velocities.end(), TV::Zero());

    // calculate Lame parameters
    mu = 0.5 * E / (1 + nu);
    lambda = E * nu / ((1 + nu)*(1 - 2*nu));
}

template<class T, int dim>
Eigen::Matrix<T,dim,dim> FemSimulation<T,dim>::linearPiola(TM F){
    // P(F) = mu * (F + F^T - 2I) + lambda * tr(F - I)*I
    TM P = mu*(F + F.transpose() - 2*TM::Identity()) + lambda * (F.trace() - dim) * TM::Identity();
    return P;
}

template<class T, int dim>
void FemSimulation<T,dim>::writeFrame(int frameNum){
    // valid for 2D and 3D
    Partio::ParticlesDataMutable *parts = Partio::create();
    Partio::ParticleAttribute posH, mH, vH, fH;
    mH = parts->addAttribute("m", Partio::VECTOR, 1);
    vH = parts->addAttribute("v", Partio::VECTOR, dim);
    fH = parts->addAttribute("f", Partio::VECTOR, dim);
    posH = parts->addAttribute("position", Partio::VECTOR, dim);

    for (unsigned int i = 0; i < positions.size(); i++)
    {
        int idx = parts->addParticle();
        float *p = parts->dataWrite<float>(posH, idx);
        float *m = parts->dataWrite<float>(mH, idx);
        float *v = parts->dataWrite<float>(vH, idx);
        float *f = parts->dataWrite<float>(fH, idx);
        m[0] = mass[i];
        for (int k = 0; k < dim; k++){
            p[k] = positions[i](k);
            v[k] = velocities[i](k);
            f[k] = force[i](k);
        }
    }

    std::string particleFile = "../output/frame" + std::to_string(frameNum) + ".bgeo";
    Partio::write(particleFile.c_str(), *parts);
    parts->release();
    std::cout << "=====Writing Frame " << frameNum << "!=====" << std::endl;
}

template class FemSimulation<float, 2>;