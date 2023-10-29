// Copyright (c) 2023 Nicholas Pellizzari <nifrpe@gmail.com>
// Distributed under MIT License (see LICENSE file in root directory).

#include <SimpleFE1DCalc/SimpleFE1DCalc.hpp>
#include <catch2/catch_all.hpp>

/**
 * Functor to compute weak residual integrand for the one-species reaction/diffusion problem:
 *
 * \f[ R = w c \frac{\partial u}{\partial t} + \frac{\text{d}w}{\text{d}z} k \frac{\partial
 * u}{\partial z} + w r u - w f. \f]
 */
struct DiffusionReactionFunctor {

  // alias the individual fields... not useful for this case with a single field, but a useful
  // technique for larger, more complicated problems
  enum Fields : std::ptrdiff_t { u };

  double c{};
  double k{};
  double r{};
  double f{};

  template <class E1, class E2, class E3, class E4, class E5>
  [[nodiscard]] auto compute_weak_residual_integrand([[maybe_unused]] const double t,
                                                     [[maybe_unused]] const double z,
                                                     const Eigen::MatrixBase<E1>& fields,
                                                     const Eigen::MatrixBase<E2>& fields_ddz,
                                                     const Eigen::MatrixBase<E3>& fields_ddt,
                                                     const Eigen::MatrixBase<E4>& sf,
                                                     const Eigen::MatrixBase<E5>& sf_ddz) const {
    return sf * c * fields_ddt(u) + sf_ddz * k * fields_ddz(u) + sf * r * fields(u) - sf * f;
  }
};

[[nodiscard]] Eigen::Vector4d calculated_residuals(const double c,
                                                   const double k,
                                                   const double r,
                                                   const double f,
                                                   const double L,
                                                   const double u0,
                                                   const double uL) {
  constexpr std::ptrdiff_t SF_ORDER = 2;
  constexpr std::ptrdiff_t QUAD_ORDER = 2;

  const SimpleFE1DCalc::Problem<DiffusionReactionFunctor, double> problem{
      DiffusionReactionFunctor{c, k, r, f},
      Eigen::VectorXd::LinSpaced(4, 0, L),  // uniform mesh
      std::vector<SimpleFE1DCalc::EssentialBC<double>>{
          {{DiffusionReactionFunctor::u, 0, u0}, {DiffusionReactionFunctor::u, -1, uL}}}};

  const double t = 0;
  const Eigen::Vector4d nodal_field_matrix = Eigen::Vector4d::Ones();
  const Eigen::Vector4d nodal_field_time_derivative_matrix = Eigen::Vector4d::Ones();
  Eigen::Vector4d nodal_residual_matrix = Eigen::Vector4d::Zero();

  SimpleFE1DCalc::
      compute_system_residuals<DiffusionReactionFunctor, double, SF_ORDER, 1, QUAD_ORDER>(
          problem,
          t,
          nodal_field_matrix,
          nodal_field_time_derivative_matrix,
          nodal_residual_matrix);

  return nodal_residual_matrix;
}

[[nodiscard]] Eigen::Vector4d exact_residuals(const double c,
                                              const double k,
                                              const double r,
                                              const double f,
                                              const double L,
                                              const double u0,
                                              const double uL) {
  const Eigen::Vector4d nodal_field_matrix = Eigen::Vector4d::Ones();
  const Eigen::Vector4d nodal_field_time_derivative_matrix = Eigen::Vector4d::Ones();

  // compute based on an exact analytical formula for the assembled system of equations
  return (c * L / 18) * Eigen::Matrix4d{{0, 0, 0, 0}, {1, 4, 1, 0}, {0, 1, 4, 1}, {0, 0, 0, 0}} *
             nodal_field_time_derivative_matrix +
         (k / L * 3) * Eigen::Matrix4d{{0, 0, 0, 0}, {-1, 2, -1, 0}, {0, -1, 2, -1}, {0, 0, 0, 0}} *
             nodal_field_matrix +
         (r * L / 18) * Eigen::Matrix4d{{0, 0, 0, 0}, {1, 4, 1, 0}, {0, 1, 4, 1}, {0, 0, 0, 0}} *
             nodal_field_matrix -
         (f * L / 6) * Eigen::Vector4d{0, 2, 2, 0} +
         Eigen::Matrix4d{{1, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 1}} *
             nodal_field_matrix -
         Eigen::Vector4d{u0, 0, 0, uL};
}

TEST_CASE("Basic single species reaction/diffusion example", "SimpleFE1DCalc") {

  const auto c = GENERATE(-1.0, 1.0, 2.0, 3.0);
  const auto k = GENERATE(-1.0, 1.0, 2.0, 3.0);
  const auto r = GENERATE(-1.0, 1.0, 2.0, 3.0);
  const auto f = GENERATE(-1.0, 1.0, 2.0, 3.0);
  const auto L = GENERATE(1.0, 2.0, 3.0);
  const auto u0 = GENERATE(-1.0, 1.0, 2.0, 3.0);
  const auto uL = GENERATE(-1.0, 1.0, 2.0, 3.0);

  const auto calculated = calculated_residuals(c, k, r, f, L, u0, uL);
  const auto exact = exact_residuals(c, k, r, f, L, u0, uL);

  for (std::ptrdiff_t i = 0; i < 4; ++i) {
    REQUIRE_THAT(calculated(i), Catch::Matchers::WithinAbs(exact(i), 1e-12));
  }
}