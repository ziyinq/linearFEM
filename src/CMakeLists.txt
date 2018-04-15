find_package(Eigen3 REQUIRED)
add_library(eigen INTERFACE IMPORTED)

set(SOURCE_FILES main.cpp
		         FemSimulation.cpp
				 FemQuadSim.cpp)
add_fem_executable(linearFEM ${SOURCE_FILES})
target_include_directories(linearFEM SYSTEM PUBLIC ${EIGEN3_INCLUDE_DIR})
target_link_libraries(linearFEM eigen)
target_link_libraries(linearFEM partio)