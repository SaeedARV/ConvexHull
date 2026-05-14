cmake_minimum_required(VERSION 3.18)
project(convex_hull_cpp LANGUAGES CXX)

find_package(pybind11 CONFIG REQUIRED)

option(CONVEX_HULL_OPENMP "Enable OpenMP for convex_hull_cpp" ON)

pybind11_add_module(convex_hull_cpp MODULE
    cpp/src/argmax_dot.cpp
    cpp/src/householder.cpp
    cpp/bindings/pybind_module.cpp
)

target_include_directories(convex_hull_cpp PRIVATE cpp/include)
target_compile_features(convex_hull_cpp PRIVATE cxx_std_17)

if (CONVEX_HULL_OPENMP)
    find_package(OpenMP)
    if (OpenMP_CXX_FOUND)
        target_link_libraries(convex_hull_cpp PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(convex_hull_cpp PRIVATE APPROXHULL_OPENMP=1)
    endif()
endif()

set_target_properties(convex_hull_cpp PROPERTIES OUTPUT_NAME "convex_hull_cpp")

install(TARGETS convex_hull_cpp
    LIBRARY DESTINATION "."
    RUNTIME DESTINATION "."
    ARCHIVE DESTINATION "."
)
