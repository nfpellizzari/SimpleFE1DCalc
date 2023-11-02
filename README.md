# SimpleFE1DCalc

A simple C++20 library to compute the assembled residuals of a one-dimensional finite element problem, based on given
essential boundary conditions, a mesh, and a functor object to compute the (possibly nonlinear) weak residual integrand
for the problem of interest. Some features include:

* Weak residual evaluation is agnostic to the order of (Lagrangian) shape/interpolation functions used, and the global
  data structures.
* Easily increase the shape function order and integral (Gaussian-Legendre) quadrature order by modifying a template parameter.
* Can handle a system of fields/equations.
* Leverages the Eigen3 library and template programming to compile performant code.
* A single header file to include.

SimpleFE1DCalc requires the Boost math library and Eigen3.

For example, the weak residual integrand for a single-species diffusion/reaction problem like
```math
\frac{\text{d} R}{\text{d} z} = w c \frac{\partial u}{\partial t} + \frac{\text{d}w}{\text{d}z} k \frac{\partial u}{\partial z} + w r u - w f
```
would be implemented as a functor such as:

```c++
struct DiffusionReactionFunctor {

  enum Fields : std::ptrdiff_t { u };

  double c{};
  double k{};
  double r{};
  double f{};

  template <class E1, class E2, class E3, class E4, class E5, class E6>
  auto compute_weak_residual_integrand(const double t,
                                       const double z,
                                       const Eigen::MatrixBase<E1>& fields,
                                       const Eigen::MatrixBase<E2>& fields_ddz,
                                       const Eigen::MatrixBase<E3>& fields_ddt,
                                       const Eigen::MatrixBase<E4>& fields_ddt_ddz,
                                       const Eigen::MatrixBase<E5>& sf,
                                       const Eigen::MatrixBase<E6>& sf_ddz) const {
    return sf * c * fields_ddt(u) + sf_ddz * k * fields_ddz(u) + sf * r * fields(u) - sf * f;
  }
};
```

See [test.cpp](test/test.cpp) for a full example.

#### Present (and likely permanent) limitations:

* Problem must be one-dimensional in space.
* All fields must be defined at every node of the mesh and are interpolated over each element using the same shape
  functions.
* The shape functions are of the Lagrangian type.
* Only fixed-order Gaussian quadrature is supported.
* Residual Jacobian matrix cannot be calculated (although I may implement this using automatic differentiation in the
  future). The intention is to have the Jacobian approximated by finite differencing residuals evaluated using this
  library.
* Only first-order time and spatial derivatives are provided to the weak residual functor, however higher order
  derivatives could be trivially added.
* Boundary conditions other than essential (Dirichlet) have to be handled by the weak residual functor.