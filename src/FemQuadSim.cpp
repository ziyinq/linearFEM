#include "FemQuadSim.h"

template <class T, int dim>
void FemQuadSim<T, dim>::createMesh()
{
    // create linear triangle mesh
    int xdim = 10;
    int ydim = 4;

    // create quadratic triangle mesh
    // TODO: not valid for 3D
    xdim = xdim * 2 - 1;
    ydim = ydim * 2 - 1;
    T dx = width / (xdim-1);
    T dy = height / (ydim-1);

    // generate points for mesh
    for (int i = 0; i < xdim; i++)
    {
        for (int j = 0; j < ydim; j++)
        {
            TV point(i * dx, j * dy);
            positions.push_back(point);
            // set left wall to be boundary wall
            if (i == 0)
                boundaryIdx.push_back(j);
        }
    }

    // generate index for mesh
    for (int i = 0; i < xdim - 2; i = i + 2)
    {
        for (int j = 0; j < ydim - 2; j = j + 2)
        {
            int a = i * ydim + j;
            int b = i * ydim + j + 1;
            int c = i * ydim + j + 2;
            int d = (i + 1) * ydim + j + 2;
            int e = (i + 1) * ydim + j + 1;
            int f = (i + 1) * ydim + j;
            int g = (i + 2) * ydim + j + 2;
            int h = (i + 2) * ydim + j + 1;
            int k = (i + 2) * ydim + j;
            //Eigen::Matrix<int, 6, 1> first(a, f, k, e, c, b);
            //Eigen::Matrix<int, 6, 1> second(k, h, g, d, c, e);
            Eigen::Matrix<int, 6, 1> first;
            first << a, f, k, e, c, b;
            Eigen::Matrix<int, 6, 1> second;
            second << g, d, c, e, k, h;
            mesh.push_back(first);
            mesh.push_back(second);
        }
    }
    std::cout << mesh[3] << std::endl;

    auto size = positions.size();
    mass.resize(size);
    velocities.resize(size);
    force.resize(size);

    // write .obj file
    std::ofstream fs;
    std::string objFile = "../output/quadMesh.obj";
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
            fs << " " << F(i * 2) + 1;
        fs << "\n";
    }
    fs.close();
    
}

template <class T, int dim>
void FemQuadSim<T, dim>::startSimulation()
{
    std::cout << "======Simulation Starts!=====" << std::endl;
    createMesh();
    initialize();
    for (int step = 0; step < numSteps; step++)
    {
        // reset force to be gravity
        std::fill(force.begin(), force.end(), TV::Zero());

        // calculate force
        for (size_t i = 0; i < mesh.size(); i++)
        {
            TM Ds = TM::Zero();
            // quadratic triangle, we have
            // [3x_0 + 2x_1 - 2x_3 - 3x_4, 2x_1 + 3x_2 - 3x_4 - 2x_5]
            // [3y_0 + 2y_1 - 2y_3 - 3y_5, 2y_1 + 3y_2 - 3y_4 - 2y_5]
            Ds.col(0) = 3.f * positions[mesh[i](0)] + 2.f * positions[mesh[i](1)] - 2.f * positions[mesh[i](3)] - 3.f * positions[mesh[i](4)];
            Ds.col(1) = 2.f * positions[mesh[i](1)] + 3.f * positions[mesh[i](2)] - 3.f * positions[mesh[i](4)] - 2.f * positions[mesh[i](5)];

            TM F = Ds * DmInv[i];
            TM P = linearPiola(F);
            // calculate 6 force vector
            // TM DmInvT = DmInv[i].transpose();
            // TV H0 = -3.f * W[i] * P * DmInvT.col(0);
            // TV H1 = -2.f * W[i] * P * DmInvT.col(0) - 2.f * W[i] * P * DmInvT.col(1);
            // TV H2 = -3.f * W[i] * P * DmInvT.col(1);
            // TV H3 =  2.f * W[i] * P * DmInvT.col(0);
            // TV H4 =  3.f * W[i] * P * DmInvT.col(0) + 3.f * W[i] * P * DmInvT.col(1);
            //TV H5 =  2.f * W[i] * P * DmInvT.col(1);
            // force[mesh[i](0)] += H0;
            // force[mesh[i](1)] += H1;
            // force[mesh[i](2)] += H2;
            // force[mesh[i](3)] += H3;
            // force[mesh[i](4)] += H4;
            // force[mesh[i](5)] += -H0 - H1 - H2 - H3 - H4;
            TM Ha = -3.f * W[i] * P * DmInv[i].transpose();
            TM Hb =  2.f * W[i] * P * DmInv[i].transpose();
            force[mesh[i](0)] += Ha.col(0);
            force[mesh[i](2)] += Ha.col(1);
            force[mesh[i](4)] += -Ha.col(0) - Ha.col(1);
            force[mesh[i](1)] += -Hb.col(0) - Hb.col(1);
            force[mesh[i](3)] += Hb.col(0);
            force[mesh[i](5)] += Hb.col(1);
        }

        // update velocity and advect node
        advection();
        writeFrame(step);
    }
}

template <class T, int dim>
void FemQuadSim<T, dim>::advection()
{
    for (size_t i = 0; i < positions.size(); i++)
    {
        velocities[i] += dt * (force[i] / mass[i] + gravity);
    }

    // fix boundary velocity to be 0
    for (auto X : boundaryIdx)
    {
        velocities[X] = TV::Zero();
    }

    for (size_t i = 0; i < positions.size(); i++)
    {
        positions[i] += dt * velocities[i];
    }
}

template <class T, int dim>
void FemQuadSim<T, dim>::initialize()
{
    // initialize DmInv, W, velocities, mass
    // valid for 2D and 3D
    for (auto X : mesh)
    {
        TM Dm = TM::Zero();
        // quadratic triangle, we have
        // [3x_0 + 2x_1 - 2x_3 - 3x_4, 2x_1 + 3x_2 - 3x_4 - 2x_5]
        // [3y_0 + 2y_1 - 2y_3 - 3y_5, 2y_1 + 3y_2 - 3y_4 - 2y_5]
        Dm.col(0) = 3.f * positions[X(0)] + 2.f * positions[X(1)] - 2.f * positions[X(3)] - 3.f * positions[X(4)];
        Dm.col(1) = 2.f * positions[X(1)] + 3.f * positions[X(2)] - 3.f * positions[X(4)] - 2.f * positions[X(5)];
        DmInv.push_back(Dm.inverse());
        // area
        TM a = TM::Zero();
        a.col(0) = positions[X(0)] - positions[X(5)];
        a.col(1) = positions[X(3)] - positions[X(5)];
        T thisW = a.determinant();
        if (thisW < 0.f)
            thisW *= -1;
        W.push_back(thisW);

        // estimate mass
        T thismass = thisW * density / 6;
        // TODO: fix if not linear FEM
        for (int i = 0; i < 6; i++)
        {
            mass[X(i)] += thismass;
        }
    }

    // set velocity to be zero
    std::fill(velocities.begin(), velocities.end(), TV::Zero());

    // calculate Lame parameters
    mu = 0.5 * E / (1 + nu);
    lambda = E * nu / ((1 + nu) * (1 - 2 * nu));
}

template <class T, int dim>
Eigen::Matrix<T, dim, dim> FemQuadSim<T, dim>::linearPiola(TM F)
{
    // P(F) = mu * (F + F^T - 2I) + lambda * tr(F - I)*I
    TM P = mu * (F + F.transpose() - 2 * TM::Identity()) + lambda * (F.trace() - dim) * TM::Identity();
    return P;
}

template <class T, int dim>
void FemQuadSim<T, dim>::writeFrame(int frameNum)
{
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
        for (int k = 0; k < dim; k++)
        {
            p[k] = positions[i](k);
            v[k] = velocities[i](k);
            f[k] = force[i](k);
        }
    }

    std::string particleFile = "../output/quadframe" + std::to_string(frameNum) + ".bgeo";
    Partio::write(particleFile.c_str(), *parts);
    parts->release();
    std::cout << "=====Writing Frame " << frameNum << "!=====" << std::endl;
}

template class FemQuadSim<float, 2>;