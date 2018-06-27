# linearFEM

Techniqal detail please refer to the [report](report.pdf)

Compile
=================
* This simulation code is built based on camke and C++
* To compile the C++ code, create a new folder in the root directory, type 
```
cmake ..
```
then type
```
make
```
and a executable file will generated in the src folder
* modify the main.cpp to use linear or quadratic FEM simulation under different Young's modulus
* create a folder named **output**, the simulation result .bgeo files will be written in this folder

CVAE code
================ 
* the algorithm is written in python and PyTorch
* the CVAE model is defined in *CVAEcuda.py*, and *linearFEM_CVAE(l2l).py* is the framework for training linear to linear,  *linearFEM_CVAE(l2q).py* is the framework for training linear to quadratic.