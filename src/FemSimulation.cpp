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
    nodeNum = size;
    velocities.resize(size);

    // write .obj file
    std::ofstream fs;
    std::string objFile = "../output/linearMesh.obj";
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

template <class T, int dim>
void FemSimulation<T, dim>::initialize()
{
    // initialize DmInv, W, velocities, mass
    massM = Eigen::MatrixXf::Zero(nodeNum, nodeNum);
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
        thisW *= 0.5f;
        W.push_back(thisW);

        // SECOND STEP: Build Mass Matrix
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                if (i == j)
                {
                    massM(X(i), X(j)) += thisW / 6.f;
                }
                else
                {
                    massM(X(i), X(j)) += thisW / 12.f;
                }
            }
        }
    }

    // write .obj file
    std::ofstream fs;
    std::string objFile = "../linearMassMatrix.txt";
    fs.open(objFile);
    for (int i = 0; i < nodeNum; i++)
    {
        for (int j = 0; j < nodeNum; j++)
        {
            fs << massM(i, j) << " ";
        }
        fs << "\n";
    }

    // check if mass matrix is SPD
    Eigen::LLT<Eigen::MatrixXf> lltOfA(massM); // compute the Cholesky decomposition of A
    if (lltOfA.info() == Eigen::NumericalIssue)
    {
        throw std::runtime_error("Possibly non semi-positive definitie matrix!");
    }

    // pre decompose mass matrix
    ldlt.compute(massM);

    // set velocity to be zero
    std::fill(velocities.begin(), velocities.end(), TV::Zero());

    // calculate Lame parameters
    mu = 0.5 * E / (1 + nu);
    lambda = E * nu / ((1 + nu) * (1 - 2 * nu));
}

template<class T, int dim>
void FemSimulation<T,dim>::startSimulation(){
    std::cout << "======Linear Simulation Starts!=====" << std::endl;
    createMesh();
    initialize();
    for (int step = 0; step < numSteps; step++){
        // reset force to be gravity
        forceVec = Eigen::MatrixXf::Zero(nodeNum, 2);

        buildForce(step);
        advection(step);
        writeFrame(step);
        if (step == 250)
        {
            getData(step);
            // getOriginData(step);
            std::cout << " Output linear data for frame" << step << "! " << std::endl;
        }
    }
}

template <class T, int dim>
void FemSimulation<T, dim>::buildForce(int c)
{
    for (unsigned int i = 0; i < mesh.size(); i++)
    {
        TM Ds = TM::Zero();
        for (int j = 0; j < dim; j++)
        {
            Ds.col(j) = positions[mesh[i](j)] - positions[mesh[i](dim)];
        }
        TM F = Ds * DmInv[i];
        TM P = neohookeanPiola(F);
        TM H = -W[i] * P * DmInv[i].transpose();
        TV lastForce = -H.col(0) - H.col(1);

        for (int k = 0; k < 3; k++)
        {
            if (k<2)
            {
                forceVec.row(mesh[i](k)) += dt * H.col(k);
            }
            else
            {
                forceVec.row(mesh[i](k)) += dt * lastForce;
            }
        }
    }
}

template<class T, int dim>
void FemSimulation<T,dim>::advection(int c)
{
    Eigen::MatrixXf x = ldlt.solve(forceVec);

    for (size_t i = 0; i < positions.size(); i++){
        velocities[i] += x.row(i).transpose() / density;
        velocities[i] += gravity * dt;
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

template<class T, int dim>
Eigen::Matrix<T,dim,dim> FemSimulation<T,dim>::linearPiola(TM F)
{
    // P(F) = mu * (F + F^T - 2I) + lambda * tr(F - I)*I
    TM P = mu*(F + F.transpose() - 2*TM::Identity()) + lambda * (F.trace() - dim) * TM::Identity();
    return P;
}

template <class T, int dim>
Eigen::Matrix<T, dim, dim> FemSimulation<T, dim>::neohookeanPiola(TM F)
{
    // P(F) = mu * (F - F^-T) + lambda * log(J) * F^-T
    TM FinvT = F.inverse().transpose();
    TM P = mu * (F - FinvT) + lambda * log(F.determinant()) * FinvT;
    return P;
}

template<class T, int dim>
void FemSimulation<T,dim>::writeFrame(int frameNum){
    // valid for 2D and 3D
    Partio::ParticlesDataMutable *parts = Partio::create();
    Partio::ParticleAttribute posH, mH, vH, fH;
    // mH = parts->addAttribute("m", Partio::VECTOR, 1);
    vH = parts->addAttribute("v", Partio::VECTOR, dim);
    fH = parts->addAttribute("f", Partio::VECTOR, dim);
    posH = parts->addAttribute("position", Partio::VECTOR, 3);

    for (unsigned int i = 0; i < positions.size(); i++)
    {
        int idx = parts->addParticle();
        float *p = parts->dataWrite<float>(posH, idx);
        // float *m = parts->dataWrite<float>(mH, idx);
        float *v = parts->dataWrite<float>(vH, idx);
        float *f = parts->dataWrite<float>(fH, idx);
        // m[0] = mass[i];
        for (int k = 0; k < dim; k++){
            p[k] = positions[i](k);
            v[k] = velocities[i](k);
            f[k] = forceVec.row(i).transpose()(k);
        }
        p[2] = 0;
    }

    std::string particleFile = "../output/frame" + std::to_string(frameNum) + ".bgeo";
    Partio::write(particleFile.c_str(), *parts);
    parts->release();
    // std::cout << "=====Writing Frame " << frameNum << "!=====" << std::endl;
}

template <class T, int dim>
void FemSimulation<T, dim>::getOriginData(int c)
{
    std::ofstream fs;
    std::string objFile = "../data/linear/origin/linearFrame" + std::to_string(c) + "E" + std::to_string((int)E) + ".txt";
    fs.open(objFile.c_str());
    for (auto p : positions)
    {
        fs << p(0) << " " << p(1) << std::endl;
    }
    fs.close();
}

template <class T, int dim>
void FemSimulation<T, dim>::getData(int c)
{
    int xdim = 10;
    int ydim = 4;
    // data
    std::ofstream fs;
    // std::string objFile = "../data/linear/training/linearFrame" + std::to_string(c) + "E" + std::to_string((int)E) + ".txt";
    std::string objFile = "../data/linear/test/linearFrame" + std::to_string(c) + "E" + std::to_string((int)E) + ".txt";
    fs.open(objFile.c_str());
    for (int i = 0; i < xdim; i++)
    {
        for (int j = 0; j < ydim; j++)
        {
            int index = i * ydim + j;
            fs << positions[index](0) << " " << positions[index](1) << std::endl;
            if (j != ydim-1)
            {
                TV midPos = 0.5f * positions[index] + 0.5f * positions[index + 1];
                fs << midPos(0) << " " << midPos(1) << std::endl;
            }
        }
        if ( i != xdim-1)
        {
            for (int k = 0; k < ydim; k++)
            {
                int index = i * ydim + k;
                int midIdx = index + ydim;
                int crossIdx = midIdx + 1;
                TV midPos = 0.5f * positions[index] + 0.5f * positions[midIdx];
                fs << midPos(0) << " " << midPos(1) << std::endl;
                if (k != ydim -1)
                {
                    TV crossPos = 0.5f * positions[index] + 0.5f * positions[crossIdx];
                    fs << crossPos(0) << " " << crossPos(1) << std::endl;
                }
            }
        }
    }
    fs.close();
}

template class FemSimulation<float, 2>;