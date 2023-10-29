add_library(SimpleFE1DCalc INTERFACE SimpleFE1DCalc/SimpleFE1DCalc.hpp)

target_include_directories(SimpleFE1DCalc INTERFACE .)

target_compile_features(SimpleFE1DCalc INTERFACE cxx_std_20)

target_link_libraries(SimpleFE1DCalc INTERFACE Eigen3::Eigen Boost::math_tr1)