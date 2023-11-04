// Copyright (c) 2023 Nicholas Pellizzari <nifrpe@gmail.com>
// Distributed under MIT License (see LICENSE file in root directory).

#ifndef SIMPLE_FE1D_CALC_HEADER_GUARD
#define SIMPLE_FE1D_CALC_HEADER_GUARD

#include <Eigen/Core>
#include <boost/math/quadrature/gauss.hpp>
#include <boost/math/tools/polynomial.hpp>

namespace SimpleFE1DCalc {

/**
 * Basic POD type to represent an evaluation point for numerical quadrature, such that
 * \f[ I = \int_{-1}^1 f(x) \, \text{d}x \approx \sum_i f(x_i) \, w_i. \f]
 *
 * @tparam T Floating pointing type underlying calculations.
 */
template <std::floating_point T>
struct QuadraturePoint {
  /**
   * Coordinate \f$ -1 \leq x_i \leq 1 \f$ to evaluate the integrand.
   */
  T abscissa{};

  /**
   * Weight \f$ w_i \f$ for integrand evaluation at the point.
   */
  T weight{};
};

/**
 * Get a list of Gaussian or Gauss-Legendre quadrature points to approximate an integral, which can
 * exactly integrate a polynomial of order \f$ \leq 2N - 1 \f$.
 *
 * This function repackages point data from boost::math::quadrature.
 *
 * @tparam N Gaussian quadrature number of evaluations for integration.
 * @tparam T Floating pointing type underlying calculations.
 *
 * @return Array of points at which to evaluate the integrand.
 *
 * @see QuadraturePoint
 */
template <std::ptrdiff_t N, std::floating_point T>
[[nodiscard]] static const std::array<QuadraturePoint<T>, N>& get_Gaussian_quadrature_points() {
  // alias the boost library type
  using boost::math::quadrature::detail::gauss_constant_category;
  using gauss_detail =
      boost::math::quadrature::detail::gauss_detail<T, N, gauss_constant_category<T>::value>;

  // compute the set of points
  static const auto points = []() {
    std::array<QuadraturePoint<T>, N> points{};

    // get half-plane points from boost
    const auto& abscissae = gauss_detail::abscissa();
    const auto& weights = gauss_detail::weights();

    // reorganize the points the way we want them
    constexpr std::ptrdiff_t start =
        N % 2 == 1 ? 1 : 0;  // points are located symmetrically w.r.t. \f$ x = 0 \f$, but for odd N
    // there will be a point at \f$ x = 0 \f$.
    if constexpr (start == 1) { points[0] = {abscissae[0], weights[0]}; }
    for (std::ptrdiff_t i = start; i < abscissae.size(); ++i) {
      points[2 * (i - start) + start] = {abscissae[i], weights[i]};
      points[2 * (i - start) + start + 1] = {-abscissae[i], weights[i]};
    }

    return points;
  }();

  return points;
}

/**
 * Compute all Lagrangian shape functions of a given order at a particular point.
 *
 * @tparam T Floating pointing type underlying calculations.
 * @tparam SF_ORDER Number of nodes constituting Lagrangian element.
 * @tparam D Order of derivative of shape functions to compute (default = 0, i.e. no derivative).
 *
 * @param r Coordinate in the domain -1 <= r <= 1.
 *
 * @return Eigen vector containing values of each shape function in the set.
 */
template <std::floating_point T, std::ptrdiff_t SF_ORDER, std::ptrdiff_t D = 0>
[[nodiscard]] static Eigen::Matrix<T, SF_ORDER, 1> compute_sf(const T r) {
  // failure may not be obvious if these assertions aren't met
  static_assert(D >= 0);
  static_assert(SF_ORDER >= 2);

  // construct the set of polynomials for the given D and SF_ORDER
  using polynomial = boost::math::tools::polynomial<T>;
  static std::array<polynomial, SF_ORDER> polynomials = [&]() {
    // divide the domain -1 <= r <= 1
    const Eigen::Matrix<T, SF_ORDER, 1> mesh =
        Eigen::Matrix<T, SF_ORDER, 1>::LinSpaced(SF_ORDER, -1.0, 1.0);

    // compute the "master" Lagrangian polynomial
    const polynomial master = [&mesh]() {
      polynomial master{{1.0}};
      for (const auto z : mesh) { master *= polynomial{{-z, 1.0}}; }
      return master;
    }();

    // compute the individual polynomials
    std::array<polynomial, SF_ORDER> polynomials{};
    for (std::ptrdiff_t i = 0; i < SF_ORDER; ++i) {
      polynomials[i] = master / polynomial{{-mesh[i], 1.0}};
      polynomials[i] /= polynomials[i](mesh[i]);
      for (std::ptrdiff_t d = 0; d < D; ++d) { polynomials[i] = polynomials[i].prime(); }
    }
    return polynomials;
  }();

  // evaluate the polynomials and return
  Eigen::Matrix<T, SF_ORDER, 1> evaluated{};
  std::transform(
      polynomials.cbegin(), polynomials.cend(), evaluated.begin(), [r](const polynomial& p) {
        return p(r);
      });
  return evaluated;
}

/**
 * Computes the finite element residuals for a single element, by iterating over a set of Gaussian
 * quadrature points and evaluating the given problem functor.
 *
 * @tparam F Type of functor to compute weak form residuals for element.
 * @tparam T Floating pointing type underlying calculations.
 * @tparam QUAD_ORDER Gaussian quadrature number of evaluations for integration.
 * @tparam SF_ORDER Number of nodes constituting Lagrangian element.
 * @tparam E1 Eigen template for mesh
 * @tparam E2 Eigen template for dof_matrix
 * @tparam E3 Eigen template for dof_derivative_matrix
 * @tparam E4 Eigen template for residual_matrix
 *
 * @param t Current simulation time for evaluation.
 * @param mesh (SF_ORDER x 1) matrix containing node 1D coordinates.
 * @param nodal_field_matrix (SF_ORDER x N_FIELDS) matrix containing field values at the nodes
 *                           constituting a particular element.
 * @param nodal_field_time_derivative_matrix (SF_ORDER x N_FIELDS) matrix containing field time
 *                                           derivative values at the nodes.
 * @param nodal_residual_matrix (SF_ORDER x N_FIELDS) mutable matrix to add element nodal
 * residuals into, corresponding to the values in `nodal_field_matrix`.
 * @param functor Functor to compute weak form residuals for element.
 */
template <class F,
          std::floating_point T,
          std::ptrdiff_t QUAD_ORDER,
          std::ptrdiff_t SF_ORDER,
          class E1,
          class E2,
          class E3,
          class E4>
static void compute_element_residuals(
    const T t,
    const Eigen::MatrixBase<E1>& mesh,
    const Eigen::MatrixBase<E2>& nodal_field_matrix,
    const Eigen::MatrixBase<E3>& nodal_field_time_derivative_matrix,
    Eigen::MatrixBase<E4>& nodal_residual_matrix,
    const F& functor) {
  for (const auto& point : get_Gaussian_quadrature_points<QUAD_ORDER, T>()) {
    const auto sf = compute_sf<T, SF_ORDER>(point.abscissa);
    const auto z = sf.dot(mesh);

    const auto sf_ddr = compute_sf<T, SF_ORDER, 1>(point.abscissa);
    const auto z_ddr = sf_ddr.dot(mesh);
    const auto sf_ddz = sf_ddr / z_ddr;

    const auto w = point.weight * z_ddr;

    const auto fields = nodal_field_matrix.transpose() * sf;
    const auto fields_ddz = nodal_field_matrix.transpose() * sf_ddz;
    const auto fields_ddt = nodal_field_time_derivative_matrix.transpose() * sf;
    const auto fields_ddt_ddz = nodal_field_time_derivative_matrix.transpose() * sf_ddz;

    nodal_residual_matrix +=
        w * functor.compute_weak_residual_integrand(
                t, z, fields, fields_ddz, fields_ddt, fields_ddt_ddz, sf, sf_ddz);
  }
}

/**
 * Basic POD type to represent an essential (Dirichlet) boundary condition.
 *
 * @tparam T Floating pointing type underlying calculations.
 */
template <std::floating_point T>
struct EssentialBC {
  std::ptrdiff_t field_index{}; /**< Index of field which condition applies to. */
  std::ptrdiff_t node_index{};  /**< Index of node which condition applies to; may be negative, e.g.
                                   -1 corresponds to the last node in the mesh. */
  T value{};                    /**< Value which field should be set to at the node. */
};

template <class F, std::floating_point T>
struct Problem {
  const F& functor{};
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& mesh{};
  const std::vector<EssentialBC<T>>& essential_bcs{};
};

/**
 * Computes the finite element residuals for an assembly of elements spanning the given mesh, by
 * appending the results of calls to `compute_element_residuals`, and then applying the given set
 * of essential boundary conditions.
 *
 * @tparam F Type of functor to compute weak form residuals for element.
 * @tparam T Floating pointing type underlying calculations.
 * @tparam SF_ORDER Number of nodes constituting Lagrangian element.
 * @tparam N_FIELDS Number of fields at each node.
 * @tparam QUAD_ORDER Gaussian quadrature number of evaluations for integration.
 * @tparam E1 Eigen template for mesh
 * @tparam E2 Eigen template for dof_matrix
 * @tparam E3 Eigen template for dof_derivative_matrix
 *
 * @param t Current simulation time for evaluation.
 * @param mesh (N_NODES x 1) matrix containing node 1D coordinates.
 * @param nodal_field_matrix (N_NODES x N_FIELDS) matrix containing field values at the nodes
 *                           constituting a particular element.
 * @param nodal_field_time_derivative_matrix (N_NODES x N_FIELDS) matrix containing field time
 *                                           derivative values at the nodes.
 * @param nodal_residual_matrix (N_NODES x N_FIELDS) mutable matrix to add nodal residuals into,
 *                              corresponding to the values in `nodal_field_matrix`.
 * @param functor Functor to compute weak form residuals for element.
 */
template <class F,
          std::floating_point T,
          std::ptrdiff_t SF_ORDER,
          std::ptrdiff_t N_FIELDS,
          std::ptrdiff_t QUAD_ORDER,
          class E1,
          class E2,
          class E3>
static void compute_system_residuals(
    const Problem<F, T>& problem,
    const T t,
    const Eigen::MatrixBase<E1>& nodal_field_matrix,
    const Eigen::MatrixBase<E2>& nodal_field_time_derivative_matrix,
    Eigen::MatrixBase<E3>& nodal_residual_matrix) {
  const std::ptrdiff_t N_NODES = problem.mesh.size();
  const std::ptrdiff_t N_ELEMENTS = (N_NODES - 1) / (SF_ORDER - 1);

  for (std::ptrdiff_t e = 0; e < N_ELEMENTS; ++e) {
    const auto LH_NODE = e * (SF_ORDER - 1);

    const auto element_mesh = problem.mesh.template segment<SF_ORDER>(LH_NODE);

    const auto element_nodal_field_matrix =
        nodal_field_matrix.template block<SF_ORDER, N_FIELDS>(LH_NODE, 0);

    const auto element_nodal_field_time_derivative_matrix =
        nodal_field_time_derivative_matrix.template block<SF_ORDER, N_FIELDS>(LH_NODE, 0);

    auto element_nodal_residual_matrix =
        nodal_residual_matrix.template block<SF_ORDER, N_FIELDS>(LH_NODE, 0);

    compute_element_residuals<F, T, QUAD_ORDER, SF_ORDER>(
        t,
        element_mesh,
        element_nodal_field_matrix,
        element_nodal_field_time_derivative_matrix,
        element_nodal_residual_matrix,
        problem.functor);
  }

  for (const auto& BC : problem.essential_bcs) {
    const auto node_index = BC.node_index >= 0 ? BC.node_index : N_NODES + BC.node_index;
    nodal_residual_matrix(node_index, BC.field_index) =
        nodal_field_matrix(node_index, BC.field_index) - BC.value;
  }
}

}  // namespace SimpleFE1DCalc

#endif
