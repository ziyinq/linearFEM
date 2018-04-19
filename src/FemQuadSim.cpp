#include "FemQuadSim.h"

template <class T, int dim>
void FemQuadSim<T, dim>::createOneMesh()
{
    positions.push_back(TV(0, 0));
    positions.push_back(TV(0.5f, 0));
    positions.push_back(TV(1.f, 0));
    positions.push_back(TV(0.75f, 0.5f));
    positions.push_back(TV(0.5f, 1.f));
    positions.push_back(TV(0.25f, 0.5f));

    Eigen::Matrix<int, 6, 1> first;
    first << 0, 1, 2, 3, 4, 5;
    mesh.push_back(first);

    auto size = positions.size();
    nodeNum = size;
    velocities.resize(size);

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
            Eigen::Matrix<int, 6, 1> first;
            Eigen::Matrix<int, 6, 1> second;
            first << a, f, k, e, c, b;
            second << g, d, c, e, k, h;
            mesh.push_back(first);
            mesh.push_back(second);
        }
    }
    auto size = positions.size();
    nodeNum = size;
    velocities.resize(size);

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
void FemQuadSim<T, dim>::initialize()
{
    /// FIRST STEP: precompute Dm, H at quadrature points
    /// N1=2k^2-k, N2=2e^2-2, N3=1-3k-3e+4ke+2e^2+2k^2, N4=4ke, N5=4e-4ke-4e^2, N6=4k-4ke-4k^2 
    // H evaluated at 3 quadrature points
    // H1 = [5/3 2/3    0 -2/3 1/3   -2]^T
    //      [0   8/3 -1/3    0 1/3 -8/3]
    // H2 = [-1/3 8/3    0 -8/3 1/3    0]^T
    //      [0    2/3  5/3   -2 1/3 -2/3]
    // H3 = [-1/3 2/3    0 -2/3 -5/3    2]^T
    //    = [0    2/3 -1/3    2 -5/3 -2/3]
    H1 = TH::Zero();
    H2 = TH::Zero();
    H3 = TH::Zero();
    // H1.col(0) << 5.f / 3.f, 2.f / 3.f, 0, -2.f / 3.f, 1.f / 3.f, -2.f;
    // H1.col(1) << 0, 8.f / 3.f, -1.f / 3.f, 0, 1.f / 3.f, -8.f / 3.f;
    // H2.col(0) << -1.f / 3.f, 8.f / 3.f, 0, -8.f / 3.f, 1.f / 3.f, 0;
    // H2.col(1) << 0, 2.f / 3.f, 5.f / 3.f, -2.f, 1.f / 3.f, -2.f / 3.f;
    // H3.col(0) << -1.f / 3.f, 2.f / 3.f, 0, -2.f / 3.f, -5.f / 3.f, 2.f;
    // H3.col(1) << 0, 2.f / 3.f, -1.f / 3.f, 2.f, -5.f / 3.f, -2.f / 3.f;
    H1 = computeH(2.f/3.f, 1.f/6.f);
    H2 = computeH(1.f/6.f, 2.f/3.f);
    H3 = computeH(1.f/6.f, 1.f/6.f);

    // shape function evaluated at 6 nodes for 3 quadrature points
    // Na = [2/9 4/9 -1/9 1/9 -1/9 4/9]
    // Nb = [-1/9 4/9 2/9 4/9 -1/9 1/9]
    // Nc = [-1/9 1/9 -1/9 4/9 2/9 4/9]
    // TN N = TN::Zero();
    // N.col(0) << 2.f / 9.f, 4.f / 9.f, -1.f / 9.f, 1.f / 9.f, -1.f / 9.f, 4.f / 9.f;
    // N.col(1) << -1.f / 9.f, 4.f / 9.f, 2.f / 9.f, 4.f / 9.f, -1.f / 9.f, 1.f / 9.f;
    // N.col(2) << -1.f / 9.f, 1.f / 9.f, -1.f / 9.f, 4.f / 9.f, 2.f / 9.f, 4.f / 9.f;

    Eigen::Matrix<T, 6, 3> xw = Eigen::Matrix<T, 6, 3>::Zero();
    xw.row(0) << 0.44594849091597, 0.44594849091597, 0.22338158967801;
    xw.row(1) << 0.44594849091597, 0.10810301816807, 0.22338158967801;
    xw.row(2) << 0.10810301816807, 0.44594849091597, 0.22338158967801;
    xw.row(3) << 0.09157621350977, 0.09157621350977, 0.10995174365532;
    xw.row(4) << 0.09157621350977, 0.81684757298046, 0.10995174365532;
    xw.row(5) << 0.81684757298046, 0.09157621350977, 0.10995174365532;

    for (int i = 0; i < 6; i++)
    {
        TH thisH = computeH(xw(i,0), xw(i,1));
        TMassN thisN = computeN(xw(i,0), xw(i,1));
        massH.push_back(thisH);
        massN.push_back(thisN);
    }

    // initialize mass with 0
    massM = Eigen::MatrixXf::Zero(nodeNum, nodeNum);
    // initialize DmInv, W, velocities, mass
    // valid for 2D and 3D
    for (auto X : mesh)
    {
        TD Dm = TD::Zero();
        for (unsigned int i = 0; i < 6; i++)
        {
            Dm.col(i) = positions[X(i)];
        }
        TM DmH1 = Dm * H1;
        TM DmH2 = Dm * H2;
        TM DmH3 = Dm * H3;

        T DmH1Det = DmH1.determinant();
        T DmH2Det = DmH2.determinant();
        T DmH3Det = DmH3.determinant();

        DmHInv.push_back(DmH1.inverse());
        DmHInv.push_back(DmH2.inverse());
        DmHInv.push_back(DmH3.inverse());

        DmHDet.push_back(DmH1Det);
        DmHDet.push_back(DmH2Det);
        DmHDet.push_back(DmH3Det);

        // SECOND STEP: Build Mass Matrix
        for (int i = 0; i < 6; i++)
        {
            for (int j = 0; j < 6; j++)
            {
                T thisMass = 0.f;
                for (int k = 0; k < 6; k++)
                {
                    TM massDmH = Dm * massH[k];
                    thisMass += massN[k](i) * massN[k](j) * massDmH.determinant() * xw(k,2);
                }
                massM(X(i), X(j)) += thisMass;
            }
        }
    }

    // write .obj file
    std::ofstream fs;
    std::string objFile = "../quadMassMatrix.txt";
    fs.open(objFile);
    for (int i = 0; i < nodeNum; i++)
    {
        for (int j = 0; j < nodeNum; j++)
        {
            fs << massM(i,j) << " ";
        }
        fs << "\n";
    }

    Eigen::LLT<Eigen::MatrixXf> lltOfA(massM); // compute the Cholesky decomposition of A
    if (lltOfA.info() == Eigen::NumericalIssue)
    {
        throw std::runtime_error("Possibly non semi-positive definitie matrix!");
    }

    // set velocity to be zero
    std::fill(velocities.begin(), velocities.end(), TV::Zero());

    // calculate Lame parameters
    mu = 0.5 * E / (1 + nu);
    lambda = E * nu / ((1 + nu) * (1 - 2 * nu));
}

template <class T, int dim>
void FemQuadSim<T, dim>::startSimulation()
{
    std::cout << "======Simulation Starts!=====" << std::endl;
    createMesh();
    // createOneMesh();
    initialize();
    for (int step = 0; step < numSteps; step++)
    {
        // reset force to be 0
        forceVec = Eigen::MatrixXf::Zero(nodeNum, 2);

        // calculate force
        buildForce(step);

        // update velocity and advect node
        advection(step);
        writeFrame(step);
    }
}

template <class T, int dim>
void FemQuadSim<T, dim>::buildForce(int c)
{
    for (unsigned int i = 0; i < mesh.size(); i++)
    {
        TD Ds = TD::Zero();
        for (unsigned int j = 0; j < 6; j++)
        {
            Ds.col(j) = positions[mesh[i](j)];
        }
        TM DsH1 = Ds * H1;
        TM DsH2 = Ds * H2;
        TM DsH3 = Ds * H3;

        TM F1 = DsH1 * DmHInv[i * 3];
        TM F2 = DsH2 * DmHInv[i * 3 + 1];
        TM F3 = DsH3 * DmHInv[i * 3 + 2];

        TM P1 = neohookeanPiola(F1);
        TM P2 = neohookeanPiola(F2);
        TM P3 = neohookeanPiola(F3);

        // fi += sum_g Pg(DmH)^-T(dN/dksi)^T*|DmH|g*1/3
        TD thisForce = - dt *(P1 * DmHInv[i * 3].transpose()     * H1.transpose() * DmHDet[i * 3] 
                       + P2 * DmHInv[i * 3 + 1].transpose() * H2.transpose() * DmHDet[i * 3 + 1] 
                       + P3 * DmHInv[i * 3 + 2].transpose() * H3.transpose() * DmHDet[i * 3 + 2]) / 3.f;

        for (int k = 0; k < 6; k++)
        {
            forceVec.row(mesh[i](k)) += thisForce.col(k);
        }
    }
}

template <class T, int dim>
void FemQuadSim<T, dim>::advection(int c)
{

    Eigen::LDLT<Eigen::MatrixXf> ldlt;
    ldlt.compute(massM);
    Eigen::MatrixXf x = ldlt.solve(forceVec);

    for (size_t i = 0; i < positions.size(); i++)
    {
        velocities[i] += x.row(i).transpose() / density;
        velocities[i] += gravity * dt;
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
Eigen::Matrix<T, 6, dim> FemQuadSim<T, dim>::computeH(T x, T y)
{
    TH thisH = TH::Zero();
    thisH(0,0) = 4.f*x - 1.f;
    thisH(1,0) = 4.f*y;
    thisH(1,1) = 4.f*x;
    thisH(2,1) = 4.f*y - 1.f;
    thisH(3,0) = -4.f*y;
    thisH(3,1) = 4.f - 4.f*x - 8.f*y;
    thisH(4,0) = -3.f + 4.f*y + 4.f*x;
    thisH(4,1) = -3.f + 4.f*x + 4.f*y;
    thisH(5,0) = 4.f - 4.f*y - 8.f*x;
    thisH(5,1) = -4.f * x;
    return thisH;
}

template <class T, int dim>
Eigen::Matrix<T, 6, 1> FemQuadSim<T, dim>::computeN(T x, T y)
{
    TMassN thisN = TMassN::Zero();
    thisN(0) = 2.f*x*x - x;
    thisN(1) = 4.f*x*y;
    thisN(2) = 2.f*y*y - y;
    thisN(3) = 4.f*y - 4.f*x*y - 4.f*y*y;
    thisN(4) = 1.f - 3.f*x - 3.f*y + 4.f*x*y + 2.f*x*x + 2.f*y*y;
    thisN(5) = 4.f*x - 4.f*x*y - 4.f*x*x;
    return thisN;
}

template <class T, int dim>
Eigen::Matrix<T, dim, dim> FemQuadSim<T, dim>::linearPiola(TM F)
{
    // P(F) = mu * (F + F^T - 2I) + lambda * tr(F - I)*I
    TM P = mu * (F + F.transpose() - 2 * TM::Identity()) + lambda * (F.trace() - dim) * TM::Identity();
    return P;
}

template <class T, int dim>
Eigen::Matrix<T, dim, dim> FemQuadSim<T, dim>::neohookeanPiola(TM F)
{
    // P(F) = mu * (F - F^-T) + lambda * log(J) * F^-T
    TM FinvT = F.inverse().transpose();
    TM P = mu * (F - FinvT) + lambda * log(F.determinant()) * FinvT;
    return P;
}

template <class T, int dim>
void FemQuadSim<T, dim>::writeFrame(int frameNum)
{
    // valid for 2D and 3D
    Partio::ParticlesDataMutable *parts = Partio::create();
    Partio::ParticleAttribute posH, mH, vH, fH;
    //mH = parts->addAttribute("m", Partio::VECTOR, 1);
    vH = parts->addAttribute("v", Partio::VECTOR, dim);
    fH = parts->addAttribute("f", Partio::VECTOR, dim);
    posH = parts->addAttribute("position", Partio::VECTOR, dim);

    for (unsigned int i = 0; i < positions.size(); i++)
    {
        int idx = parts->addParticle();
        float *p = parts->dataWrite<float>(posH, idx);
        //float *m = parts->dataWrite<float>(mH, idx);
        float *v = parts->dataWrite<float>(vH, idx);
        float *f = parts->dataWrite<float>(fH, idx);
        //m[0] = mass[i];
        for (int k = 0; k < dim; k++)
        {
            p[k] = positions[i](k);
            v[k] = velocities[i](k);
            f[k] = forceVec.row(i).transpose()(k);
        }
    }

    std::string particleFile = "../output/quadframe" + std::to_string(frameNum) + ".bgeo";
    Partio::write(particleFile.c_str(), *parts);
    parts->release();
    //std::cout << "=====Writing Frame " << frameNum << "!=====" << std::endl;
}

template class FemQuadSim<float, 2>;