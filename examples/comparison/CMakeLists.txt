set(CMAKE_CXX_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -O2")

find_package(acados REQUIRED)
include_directories(${ACADOS_INCLUDE_DIRS})

add_executable(mecotron_step_rtpgm src/mecotron_step_rtpgm.cpp)
add_executable(mecotron_step_pgm src/mecotron_step_pgm.cpp)
add_executable(mecotron_step_qpoases src/mecotron_step_qpoases.cpp)
add_executable(mecotron_step_hpmpc src/mecotron_step_hpmpc.cpp)

target_link_libraries(mecotron_step_qpoases acados)
target_link_libraries(mecotron_step_hpmpc acados)
