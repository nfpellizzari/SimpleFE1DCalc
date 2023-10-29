cmake_minimum_required(VERSION 3.25)

project(SimpleFE1DCalc
        VERSION 1.0.0
        LANGUAGES CXX)

find_package(Eigen3 REQUIRED)

set(Boost_USE_STATIC_LIBS ON)
set(Boost_NO_WARN_NEW_VERSIONS 1) # silence Boost warning
find_package(Boost REQUIRED COMPONENTS math_tr1)

add_subdirectory(src)

# Only build test if this is the main project
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    add_subdirectory(test)
endif()